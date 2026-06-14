"""
Flask-SocketIO event handlers for the /ecg namespace.

Server → Client events (emitted from SerialReaderService thread):
  ecg_chunk        {samples: [float], bpm: float, ts: int, session_id: int}
  beat_classified  {class_id, class_name, short_name, confidence, probabilities,
                    alert, bpm, beat_index, motion_flag, session_id, timestamp, inference_ms}
  motion_alert     {motion_flag: bool, level: float}
  system_status    {recording: bool, serial_connected: bool, model_loaded: bool, session_id}
  alert            {type: str, message: str, class_id: int}

Client → Server events:
  start_recording  {patient_id: int, notes: str}
  stop_recording   {}
  ping             {}  → replies with 'pong'
"""

import logging
from datetime import datetime
from flask import request
from flask_socketio import Namespace, emit

from app.extensions import socketio, db, registry as services
from app.models.patient import Patient
from app.models.session import RecordingSession

log = logging.getLogger(__name__)


class ECGNamespace(Namespace):
    """SocketIO namespace for real-time ECG streaming on /ecg."""

    def on_connect(self) -> None:
        log.info(f"Client connected: {request.sid}")
        reader = services.get("serial_reader")
        emit("system_status", {
            "recording": reader.is_running if reader else False,
            "serial_connected": False,
            "model_loaded": services.get("inference_engine") is not None,
            "session_id": None,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def on_disconnect(self, reason=None) -> None:
        log.info(f"Client disconnected: {request.sid}, reason={reason}")

    def on_ping(self, data: dict) -> None:
        emit("pong", {"ts": datetime.utcnow().isoformat()})

    def on_start_recording(self, data: dict) -> None:
        """
        Start a new recording session.
        data: {patient_id: int, notes: str (optional)}
        """
        patient_id = data.get("patient_id")
        if not patient_id:
            emit("error_msg", {"message": "patient_id is required"})
            return

        patient = Patient.query.get(patient_id)
        if not patient:
            emit("error_msg", {"message": f"Patient {patient_id} not found"})
            return

        reader = services.get("serial_reader")
        if reader and reader.is_running:
            emit("error_msg", {"message": "Recording already in progress"})
            return

        # Create DB session
        session = RecordingSession(
            patient_id=patient_id,
            notes=str(data.get("notes", "")).strip() or None,
            is_active=True,
        )
        db.session.add(session)
        db.session.commit()

        if reader:
            reader.start(session.id)

        log.info(f"Recording started: session={session.id}, patient={patient.name}")
        emit("system_status", {
            "recording": True,
            "session_id": session.id,
            "patient_name": patient.name,
            "serial_connected": not services.get("serial_reader") or True,
            "model_loaded": services.get("inference_engine") is not None,
        })
        emit("recording_started", {
            "session_id": session.id,
            "patient_id": patient_id,
            "patient_name": patient.name,
            "started_at": session.started_at.isoformat(),
        })

    def on_stop_recording(self, data: dict = None) -> None:
        """Stop the current recording session."""
        reader = services.get("serial_reader")
        if not reader or not reader.is_running:
            emit("error_msg", {"message": "No recording in progress"})
            return

        # Close active DB session
        active_session = RecordingSession.query.filter_by(is_active=True).first()
        session_id = None
        duration = None

        if active_session:
            active_session.is_active = False
            active_session.ended_at = datetime.utcnow()
            if active_session.started_at:
                delta = active_session.ended_at - active_session.started_at
                active_session.duration_s = delta.total_seconds()
                duration = active_session.duration_s
            session_id = active_session.id
            db.session.commit()

        reader.stop()

        log.info(f"Recording stopped: session={session_id}, duration={duration}s")
        emit("system_status", {
            "recording": False,
            "session_id": session_id,
            "serial_connected": False,
            "model_loaded": services.get("inference_engine") is not None,
        })
        emit("recording_stopped", {
            "session_id": session_id,
            "duration_s": duration,
        })

    def on_get_status(self, data: dict = None) -> None:
        """Request current system status."""
        reader = services.get("serial_reader")
        active = RecordingSession.query.filter_by(is_active=True).first()
        emit("system_status", {
            "recording": reader.is_running if reader else False,
            "session_id": active.id if active else None,
            "serial_connected": reader.is_running if reader else False,
            "model_loaded": services.get("inference_engine") is not None,
            "timestamp": datetime.utcnow().isoformat(),
        })


# Register the namespace — this is the side effect imported by app/__init__.py
socketio.on_namespace(ECGNamespace("/ecg"))
