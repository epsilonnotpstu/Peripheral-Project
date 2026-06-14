"""
Real-time ECG signal processing service.

Processing chain per incoming sample:
  raw ADC sample → rolling buffer → bandpass filter → notch filter
  → Pan-Tompkins R-peak detection → BPM calculation → beat segmentation
  → normalized beat window (187 samples) ready for TFLite inference

Designed for 125 Hz sampling rate from AD8232 + ESP8266.
Uses stateful scipy.signal.sosfilt to avoid filter transients at chunk boundaries.
"""

import collections
import time
import logging
import numpy as np
import scipy.signal as sig

log = logging.getLogger(__name__)


class SignalProcessorService:
    """
    Maintains a rolling 5-second ECG buffer and processes incoming samples.

    Call add_sample(adc_value, timestamp_ms) for each new reading.
    Returns a beat dict when a complete beat is segmented, or None otherwise.
    """

    BEAT_PRE = 90    # samples before R-peak in 187-sample window
    BEAT_POST = 97   # samples after R-peak  (90 + 97 = 187)

    def __init__(self, app_config: dict):
        self.fs = app_config.get("ECG_FS", 125)
        self.beat_len = app_config.get("ECG_BEAT_LEN", 187)
        self.buffer_size = self.fs * app_config.get("BUFFER_SECONDS", 5)  # 625

        # ── Filter design (pre-computed once at startup) ────────────────────
        bp_low = app_config.get("BP_LOW", 0.5)
        bp_high = app_config.get("BP_HIGH", 40.0)
        bp_order = app_config.get("BP_ORDER", 4)
        notch_hz = app_config.get("NOTCH_HZ", 50.0)
        notch_q = app_config.get("NOTCH_Q", 30.0)

        self._sos_bp = sig.butter(
            bp_order, [bp_low, bp_high], btype="band", fs=self.fs, output="sos"
        )
        self._b_notch, self._a_notch = sig.iirnotch(notch_hz, notch_q, fs=self.fs)

        # Filter states (maintained across chunks for continuity)
        n_sections = self._sos_bp.shape[0]
        self._zi_bp = np.zeros((n_sections, 2))
        self._zi_notch = sig.lfilter_zi(self._b_notch, self._a_notch)

        # ── Rolling buffer ──────────────────────────────────────────────────
        self._raw_buffer = collections.deque(maxlen=self.buffer_size)
        self._filt_buffer = collections.deque(maxlen=self.buffer_size)
        self._ts_buffer = collections.deque(maxlen=self.buffer_size)   # timestamps ms

        # ── R-peak tracking ─────────────────────────────────────────────────
        self._rpeak_min_dist = app_config.get("RPEAK_MIN_DIST", 50)
        self._rpeak_height_frac = app_config.get("RPEAK_HEIGHT_FRAC", 0.5)
        self._last_rpeak_global = -1000    # global sample index of last detected R-peak
        self._global_sample_idx = 0
        self._beat_idx = 0                 # sequential beat counter for this session

        # ── BPM tracking ────────────────────────────────────────────────────
        self.current_bpm: float = 0.0
        self._rr_buffer = collections.deque(maxlen=8)  # last 8 RR intervals (samples)

        # ── Chunk accumulator for batch emit ────────────────────────────────
        self._chunk_acc = []
        self._chunk_ts_acc = []

        log.info(f"SignalProcessorService initialized: fs={self.fs}Hz, buffer={self.buffer_size}s×fs")

    def add_sample(self, adc_value: float, timestamp_ms: int):
        """
        Process one incoming raw ADC sample (already normalized to [0,1]).

        Args:
            adc_value: float in [0, 1]
            timestamp_ms: millisecond timestamp from ESP8266 millis()

        Returns:
            dict if a complete beat was detected:
                {beat: np.ndarray(187,), bpm: float, filtered_chunk: list,
                 timestamps: list, r_peak_idx: int, beat_index: int}
            None otherwise
        """
        # ── Append to raw buffer ─────────────────────────────────────────────
        self._raw_buffer.append(float(adc_value))
        self._ts_buffer.append(timestamp_ms)

        # ── Online bandpass filter (stateful sosfilt, 1 sample at a time) ──
        sample_arr = np.array([[float(adc_value)]])
        filtered, self._zi_bp = sig.sosfilt(
            self._sos_bp, [float(adc_value)], zi=self._zi_bp
        )
        notched, self._zi_notch = sig.lfilter(
            self._b_notch, self._a_notch, filtered, zi=self._zi_notch
        )
        filt_val = float(notched[0])
        self._filt_buffer.append(filt_val)
        self._global_sample_idx += 1

        # ── Accumulate chunk for batch processing ───────────────────────────
        self._chunk_acc.append(filt_val)
        self._chunk_ts_acc.append(timestamp_ms)

        # ── Run R-peak detection every `fs` samples (once per second) ──────
        result = None
        if self._global_sample_idx % self.fs == 0 and len(self._filt_buffer) >= self.fs:
            result = self._detect_and_segment()

        return result

    def get_chunk(self) -> tuple:
        """
        Return and clear the accumulated sample chunk for WebSocket emission.
        Call from SerialReaderService every EMIT_CHUNK_SIZE samples.

        Returns:
            (samples: list[float], timestamps: list[int])
        """
        samples = list(self._chunk_acc)
        timestamps = list(self._chunk_ts_acc)
        self._chunk_acc.clear()
        self._chunk_ts_acc.clear()
        return samples, timestamps

    def _detect_and_segment(self):
        """
        Run Pan-Tompkins simplified R-peak detection on the last 1-second buffer.
        Returns a beat dict if a new R-peak is found since last detection, else None.
        """
        filt_arr = np.array(list(self._filt_buffer), dtype=np.float32)
        n = len(filt_arr)

        # Pan-Tompkins: differentiate → square → moving-window integration
        diff = np.diff(filt_arr, prepend=filt_arr[0])
        squared = diff ** 2
        window_len = int(0.12 * self.fs)   # 120 ms integration window
        integrated = np.convolve(squared, np.ones(window_len) / window_len, mode="same")

        if integrated.max() < 1e-6:
            return None  # flat signal — no peaks

        # Adaptive threshold
        height_thresh = self._rpeak_height_frac * np.percentile(integrated, 90)

        peaks, props = sig.find_peaks(
            integrated,
            height=height_thresh,
            distance=self._rpeak_min_dist,
        )

        if len(peaks) == 0:
            return None

        # Convert local buffer index → global sample index
        # filt_buffer deque stores indices relative to (global_sample_idx - buffer_size)
        buf_start_global = self._global_sample_idx - n

        result = None
        for local_peak in peaks:
            global_peak = buf_start_global + local_peak

            # Skip if too close to last detected peak (duplicate suppression)
            if global_peak - self._last_rpeak_global < self._rpeak_min_dist:
                continue

            # ── Update RR interval and BPM ────────────────────────────────
            if self._last_rpeak_global > 0:
                rr_samples = global_peak - self._last_rpeak_global
                self._rr_buffer.append(rr_samples)
                self._update_bpm()

            self._last_rpeak_global = global_peak

            # ── Segment beat: 90 samples before + 97 after R-peak ─────────
            beat = self._segment_beat(filt_arr, local_peak)
            if beat is None:
                continue

            self._beat_idx += 1
            result = {
                "beat": beat,
                "bpm": self.current_bpm,
                "r_peak_global": global_peak,
                "beat_index": self._beat_idx,
                "timestamp_ms": self._ts_buffer[min(local_peak, len(self._ts_buffer) - 1)],
            }

        return result

    def _segment_beat(self, filt_arr: np.ndarray, r_idx: int):
        """
        Extract a 187-sample beat window centered on the R-peak.

        Uses zero-padding at signal boundaries to maintain fixed length.
        Normalizes to [0, 1] to match MIT-BIH training data preprocessing.

        Returns np.ndarray(187,) or None if signal is all zeros.
        """
        beat = np.zeros(self.beat_len, dtype=np.float32)

        src_start = r_idx - self.BEAT_PRE
        src_end = r_idx + self.BEAT_POST    # exclusive

        # Clamp to valid array bounds
        src_start_clamped = max(0, src_start)
        src_end_clamped = min(len(filt_arr), src_end)

        dst_start = src_start_clamped - src_start  # offset into beat array
        n_copy = src_end_clamped - src_start_clamped

        beat[dst_start:dst_start + n_copy] = filt_arr[src_start_clamped:src_end_clamped]

        # Min-max normalize to [0, 1] (matches MIT-BIH preprocessing)
        beat_min = beat.min()
        beat_max = beat.max()
        span = beat_max - beat_min
        if span < 1e-8:
            return None   # flat / zero signal — skip inference
        beat = (beat - beat_min) / span

        return beat.astype(np.float32)

    def _update_bpm(self) -> None:
        """Compute BPM from median RR interval, rejecting physiologically implausible values."""
        if not self._rr_buffer:
            return

        rr_arr = np.array(list(self._rr_buffer), dtype=float)
        rr_s = rr_arr / self.fs  # convert samples → seconds

        # Filter: valid RR range 0.3–2.0 s (30–200 BPM)
        valid = rr_s[(rr_s >= 0.3) & (rr_s <= 2.0)]
        if len(valid) == 0:
            return

        median_rr = float(np.median(valid))
        self.current_bpm = round(60.0 / median_rr, 1)

    def get_filtered_window(self, n_seconds: int = 10) -> np.ndarray:
        """Return last n_seconds of filtered signal for PDF report ECG strip."""
        n = n_seconds * self.fs
        buf = list(self._filt_buffer)
        return np.array(buf[-n:] if len(buf) >= n else buf, dtype=np.float32)

    def reset(self) -> None:
        """Reset all buffers and counters — call at session start."""
        self._raw_buffer.clear()
        self._filt_buffer.clear()
        self._ts_buffer.clear()
        self._rr_buffer.clear()
        self._chunk_acc.clear()
        self._chunk_ts_acc.clear()

        n_sections = self._sos_bp.shape[0]
        self._zi_bp = np.zeros((n_sections, 2))
        self._zi_notch = sig.lfilter_zi(self._b_notch, self._a_notch)

        self._last_rpeak_global = -1000
        self._global_sample_idx = 0
        self._beat_idx = 0
        self.current_bpm = 0.0
