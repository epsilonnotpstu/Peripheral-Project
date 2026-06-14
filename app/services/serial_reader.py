"""
SerialReaderService — the integration glue.

Reads ECG (and optional motion) data from ESP8266 over serial port,
feeds it through signal processing and inference, persists to SQLite,
and emits real-time WebSocket events.

Threading model:
  - Runs as threading.Thread(daemon=True) — NOT eventlet greenlet
  - pyserial's blocking readline() would starve the eventlet event loop
  - eventlet.monkey_patch() in run.py makes socketio.emit() thread-safe
  - DB writes use 'with app.app_context():' pushed in _run()

ESP8266 serial protocol (one line per sample at 125 Hz):
  Without MPU6050: ECG,<millis_ms>,<adc_0-1023>
  With MPU6050:    ECG,<millis_ms>,<adc_0-1023>,<ax100>,<ay100>,<az100>

Where ax100 = int(accel_x_in_g * 100) to avoid float formatting overhead.
"""

import threading
import time
import logging
import json
import numpy as np
from datetime import datetime

log = logging.getLogger(__name__)


class SerialReaderService:
    """
    Manages the serial reading thread lifecycle.
    start(session_id) → background thread reads serial → emits SocketIO events
    stop() → signals thread to terminate gracefully
    """

    def __init__(self, app, socketio, app_config: dict, services: dict):
        self._app = app
        self._socketio = socketio
        self._config = app_config
        self._services = services

        self._thread: threading.Thread = None
        self._stop_event = threading.Event()
        self._session_id: int = None
        self._is_running = False

        self._emit_chunk_size = int(app_config.get("EMIT_CHUNK_SIZE", 10))
        self._sample_counter = 0
        self._beat_counter = 0

        # ECG record accumulator — saved to DB every 125 samples (1 second)
        self._record_buffer = []
        self._record_ts_start = None

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self, session_id: int) -> None:
        if self._is_running:
            log.warning("SerialReaderService already running — stop first")
            return

        self._session_id = session_id
        self._stop_event.clear()
        self._sample_counter = 0
        self._beat_counter = 0
        self._record_buffer = []
        self._record_ts_start = None

        # Reset signal processor and motion detector for fresh session
        if self._services.get("signal_processor"):
            self._services["signal_processor"].reset()
        if self._services.get("motion_detector"):
            self._services["motion_detector"].reset()

        self._thread = threading.Thread(target=self._run, daemon=True, name="serial-reader")
        self._thread.start()
        self._is_running = True
        log.info(f"SerialReaderService started for session {session_id}")

    def stop(self) -> None:
        if not self._is_running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._is_running = False
        log.info(f"SerialReaderService stopped for session {self._session_id}")

    # ── Internal thread ────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Main reader loop — runs inside daemon thread with Flask app context."""
        with self._app.app_context():
            simulate = self._config.get("SIMULATE_SERIAL", False)
            if simulate:
                self._run_simulated()
            else:
                self._run_serial()

    def _run_serial(self) -> None:
        """Read from real ESP8266 serial port with automatic reconnection."""
        import serial
        import serial.serialutil

        port = self._config.get("SERIAL_PORT", "/dev/ttyUSB0")
        baudrate = int(self._config.get("SERIAL_BAUDRATE", 115200))
        timeout = float(self._config.get("SERIAL_TIMEOUT", 1.0))
        reconnect_delay = float(self._config.get("SERIAL_RECONNECT_DELAY", 3.0))

        ser = None
        while not self._stop_event.is_set():
            try:
                if ser is None or not ser.is_open:
                    log.info(f"Opening serial port {port} @ {baudrate} baud")
                    ser = serial.Serial(port, baudrate, timeout=timeout)
                    self._emit_system_status(serial_connected=True)

                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    self._process_line(line)

            except (serial.serialutil.SerialException, OSError) as e:
                log.warning(f"Serial error: {e}. Reconnecting in {reconnect_delay}s...")
                self._emit_system_status(serial_connected=False)
                if ser:
                    try:
                        ser.close()
                    except Exception:
                        pass
                    ser = None
                self._stop_event.wait(timeout=reconnect_delay)

        if ser and ser.is_open:
            ser.close()

    def _run_simulated(self) -> None:
        """
        Simulate serial input by replaying MIT-BIH test CSV data.
        Called automatically when SIMULATE_SERIAL=True (development mode).
        """
        import os
        import pandas as pd

        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "mitbih dataset",
        )
        test_path = os.path.join(data_dir, "mitbih_test.csv")

        if not os.path.exists(test_path):
            log.error(f"Simulation file not found: {test_path}")
            return

        log.info(f"Simulator: replaying {test_path}")
        df = pd.read_csv(test_path, header=None, nrows=5000)

        # Each row is a 187-sample beat — unroll into sequential samples
        fs = self._config.get("ECG_FS", 125)
        sample_interval = 1.0 / fs
        start_ms = int(time.time() * 1000)

        row_idx = 0
        while not self._stop_event.is_set():
            row = df.iloc[row_idx % len(df)]
            samples = row.iloc[:187].values.tolist()

            for i, val in enumerate(samples):
                if self._stop_event.is_set():
                    break
                ts_ms = start_ms + self._sample_counter * int(sample_interval * 1000)
                # Synthesize serial line
                line = f"ECG,{ts_ms},{int(float(val) * 1023)}"
                self._process_line(line)
                time.sleep(sample_interval)

            row_idx += 1

    def _process_line(self, line: str) -> None:
        """Parse one serial line and route through the processing pipeline."""
        parts = line.split(",")
        if not parts or parts[0] != "ECG" or len(parts) < 3:
            return

        try:
            ts_ms = int(parts[1])
            adc_raw = int(parts[2])
            adc_norm = float(adc_raw) / 1023.0

            # Optional MPU6050 data
            if len(parts) == 6:
                ax = int(parts[3]) / 100.0
                ay = int(parts[4]) / 100.0
                az = int(parts[5]) / 100.0
                motion_det = self._services.get("motion_detector")
                if motion_det:
                    motion_det.add_sample(ax, ay, az)
                    if self._sample_counter % 25 == 0:
                        self._emit_motion_status()

        except (ValueError, IndexError) as e:
            log.debug(f"Parse error on line '{line}': {e}")
            return

        # ── Signal processing ─────────────────────────────────────────────────
        processor = self._services.get("signal_processor")
        if not processor:
            return

        beat_result = processor.add_sample(adc_norm, ts_ms)
        self._sample_counter += 1

        # ── Accumulate for DB chunk saving (every 125 samples = 1 sec) ───────
        self._record_buffer.append(adc_norm)
        if self._record_ts_start is None:
            self._record_ts_start = ts_ms

        if len(self._record_buffer) >= self._config.get("ECG_FS", 125):
            self._save_ecg_chunk()

        # ── Emit raw chunk every EMIT_CHUNK_SIZE samples (~80ms) ─────────────
        if self._sample_counter % self._emit_chunk_size == 0:
            chunk_samples, chunk_ts = processor.get_chunk()
            self._socketio.emit(
                "ecg_chunk",
                {
                    "samples": chunk_samples,
                    "bpm": processor.current_bpm,
                    "session_id": self._session_id,
                    "ts": ts_ms,
                },
                namespace="/ecg",
            )

        # ── Run inference on detected beat ────────────────────────────────────
        if beat_result is not None:
            self._run_inference(beat_result)

    def _run_inference(self, beat_result: dict) -> None:
        """Run TFLite inference and emit/persist the classification result."""
        engine = self._services.get("inference_engine")
        motion_det = self._services.get("motion_detector")

        motion_flag = motion_det.is_motion_artifact() if motion_det else False
        beat = beat_result["beat"]

        if engine is None:
            # No model loaded — emit placeholder
            prediction = {
                "class_id": 0,
                "class_name": "Normal",
                "short_name": "N",
                "confidence": 0.0,
                "probabilities": [1.0, 0.0, 0.0, 0.0, 0.0],
                "alert": False,
                "inference_ms": 0.0,
            }
        else:
            try:
                prediction = engine.predict(beat)
            except Exception as e:
                log.error(f"Inference error: {e}")
                return

        self._beat_counter += 1

        # ── Persist to DB ─────────────────────────────────────────────────────
        self._save_prediction(beat_result, prediction, motion_flag)

        # ── Emit to frontend ──────────────────────────────────────────────────
        payload = {
            "class_id": prediction["class_id"],
            "class_name": prediction["class_name"],
            "short_name": prediction["short_name"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"],
            "alert": prediction["alert"],
            "bpm": beat_result["bpm"],
            "beat_index": self._beat_counter,
            "motion_flag": motion_flag,
            "session_id": self._session_id,
            "timestamp": beat_result["timestamp_ms"],
            "inference_ms": prediction["inference_ms"],
        }
        self._socketio.emit("beat_classified", payload, namespace="/ecg")

        if prediction["alert"]:
            self._socketio.emit(
                "alert",
                {
                    "type": prediction["class_name"].lower(),
                    "message": f"Alert: {prediction['class_name']} beat detected "
                               f"(confidence {prediction['confidence']*100:.1f}%)",
                    "class_id": prediction["class_id"],
                },
                namespace="/ecg",
            )

    def _save_prediction(self, beat_result: dict, prediction: dict, motion_flag: bool) -> None:
        """Persist prediction to database inside existing app context."""
        from app.extensions import db
        from app.models.prediction import Prediction

        try:
            probs = prediction["probabilities"]
            record = Prediction(
                session_id=self._session_id,
                timestamp=float(beat_result["timestamp_ms"]),
                beat_index=self._beat_counter,
                class_id=prediction["class_id"],
                class_name=prediction["class_name"],
                confidence=prediction["confidence"],
                prob_n=probs[0] if len(probs) > 0 else None,
                prob_s=probs[1] if len(probs) > 1 else None,
                prob_v=probs[2] if len(probs) > 2 else None,
                prob_f=probs[3] if len(probs) > 3 else None,
                prob_q=probs[4] if len(probs) > 4 else None,
                bpm=beat_result["bpm"],
                motion_flag=motion_flag,
                alert_raised=prediction["alert"],
            )
            db.session.add(record)
            db.session.commit()
        except Exception as e:
            log.error(f"DB prediction save error: {e}")
            db.session.rollback()

    def _save_ecg_chunk(self) -> None:
        """Save 1-second ECG chunk to ECGRecord table."""
        from app.extensions import db
        from app.models.ecg_record import ECGRecord
        from app.services.signal_processor import SignalProcessorService

        processor = self._services.get("signal_processor")
        motion_det = self._services.get("motion_detector")

        try:
            record = ECGRecord(
                session_id=self._session_id,
                timestamp=float(self._record_ts_start or 0),
                bpm=processor.current_bpm if processor else None,
                motion_flag=motion_det.is_motion_artifact() if motion_det else False,
            )
            record.set_samples(self._record_buffer)
            db.session.add(record)
            db.session.commit()
        except Exception as e:
            log.error(f"DB ECG chunk save error: {e}")
            db.session.rollback()
        finally:
            self._record_buffer = []
            self._record_ts_start = None

    def _emit_system_status(self, serial_connected: bool) -> None:
        self._socketio.emit(
            "system_status",
            {
                "recording": self._is_running,
                "serial_connected": serial_connected,
                "session_id": self._session_id,
                "model_loaded": self._services.get("inference_engine") is not None,
            },
            namespace="/ecg",
        )

    def _emit_motion_status(self) -> None:
        motion_det = self._services.get("motion_detector")
        if not motion_det or not motion_det.has_sensor():
            return
        self._socketio.emit(
            "motion_alert",
            {
                "motion_flag": motion_det.is_motion_artifact(),
                "level": motion_det.motion_level(),
            },
            namespace="/ecg",
        )
