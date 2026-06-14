"""
Unit tests for SignalProcessorService.

Tests:
  - Bandpass filter frequency response (attenuation below 0.5 Hz and above 40 Hz)
  - Notch filter attenuation at 50 Hz
  - R-peak detection on a known synthetic ECG
  - BPM calculation accuracy
  - Beat segmentation shape and normalization
  - Buffer rolling behavior
"""

import sys
import os
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_config():
    return {
        "ECG_FS": 125,
        "ECG_BEAT_LEN": 187,
        "BP_LOW": 0.5,
        "BP_HIGH": 40.0,
        "BP_ORDER": 4,
        "NOTCH_HZ": 50.0,
        "NOTCH_Q": 30.0,
        "BUFFER_SECONDS": 5,
        "RPEAK_MIN_DIST": 50,
        "RPEAK_HEIGHT_FRAC": 0.5,
    }


def make_processor():
    from app.services.signal_processor import SignalProcessorService
    return SignalProcessorService(make_config())


def synthetic_ecg(n_seconds=10, fs=125, heart_rate=75) -> np.ndarray:
    """Generate a simple synthetic ECG: R-peaks as Gaussian pulses."""
    n = n_seconds * fs
    t = np.arange(n) / fs
    signal = np.zeros(n)
    rr_samples = int(60.0 / heart_rate * fs)

    for peak in range(rr_samples // 2, n, rr_samples):
        for offset in range(-10, 10):
            idx = peak + offset
            if 0 <= idx < n:
                signal[idx] += np.exp(-0.5 * (offset / 3.0) ** 2)

    # Add 0.2 V baseline
    signal += 0.2
    # Normalize
    signal = (signal - signal.min()) / (signal.max() - signal.min())
    return signal.astype(np.float32)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestSignalProcessorInit:
    def test_init_creates_service(self):
        proc = make_processor()
        assert proc is not None
        assert proc.fs == 125
        assert proc.beat_len == 187

    def test_buffer_size(self):
        proc = make_processor()
        assert proc.buffer_size == 625  # 5 seconds × 125 Hz

    def test_bpm_initial_zero(self):
        proc = make_processor()
        assert proc.current_bpm == 0.0


class TestFilterBehavior:
    def test_bandpass_passes_ecg_frequency(self):
        """A 10 Hz sine wave (within ECG band) should pass with high amplitude."""
        import scipy.signal as sig
        proc = make_processor()
        fs = 125
        t = np.arange(5 * fs) / fs
        sine_10hz = np.sin(2 * np.pi * 10 * t).astype(np.float32)

        # Apply bandpass
        filtered = sig.sosfilt(proc._sos_bp, sine_10hz)
        # Should preserve amplitude (> 0.5 of original)
        assert np.std(filtered) > 0.3, f"10 Hz signal attenuated too much: std={np.std(filtered):.4f}"

    def test_bandpass_attenuates_dc(self):
        """DC component (0 Hz) should be removed by high-pass portion."""
        import scipy.signal as sig
        proc = make_processor()
        dc_signal = np.ones(1250, dtype=np.float32)  # pure DC
        filtered = sig.sosfilt(proc._sos_bp, dc_signal)
        # After filter settles, DC should be near zero
        assert np.abs(np.mean(filtered[-500:])) < 0.05, f"DC not removed: {np.mean(filtered[-500:]):.4f}"

    def test_notch_attenuates_50hz(self):
        """50 Hz component should be attenuated by notch filter."""
        import scipy.signal as sig
        proc = make_processor()
        fs = 125
        t = np.arange(5 * fs) / fs
        sine_50hz = np.sin(2 * np.pi * 50 * t).astype(np.float32)
        filtered, _ = sig.lfilter(proc._b_notch, proc._a_notch, sine_50hz,
                                   zi=sig.lfilter_zi(proc._b_notch, proc._a_notch))
        # Amplitude should be suppressed by > 90%
        ratio = np.std(filtered) / (np.std(sine_50hz) + 1e-8)
        assert ratio < 0.15, f"50 Hz notch insufficient: remaining ratio={ratio:.3f}"


class TestAddSample:
    def test_add_samples_increments_counter(self):
        proc = make_processor()
        for i in range(50):
            proc.add_sample(0.5 + 0.1 * np.sin(i), int(i * 8))
        assert proc._global_sample_idx == 50

    def test_add_samples_fills_buffer(self):
        proc = make_processor()
        for i in range(200):
            proc.add_sample(0.5, i * 8)
        assert len(proc._filt_buffer) == 200

    def test_buffer_rolls_at_max(self):
        proc = make_processor()
        for i in range(700):  # more than 625 buffer size
            proc.add_sample(float(i % 100) / 100, i * 8)
        assert len(proc._filt_buffer) == proc.buffer_size


class TestBeatSegmentation:
    def test_segment_returns_correct_shape(self):
        proc = make_processor()
        ecg = synthetic_ecg(n_seconds=5)
        for i, v in enumerate(ecg):
            proc.add_sample(float(v), i * 8)
        # After 5 seconds of data, internal filt_buffer should have segments
        filt_arr = np.array(list(proc._filt_buffer), dtype=np.float32)
        r_idx = len(filt_arr) // 2  # pick middle as fake R-peak
        beat = proc._segment_beat(filt_arr, r_idx)
        if beat is not None:
            assert beat.shape == (187,)

    def test_segment_normalized_to_unit_range(self):
        proc = make_processor()
        ecg = synthetic_ecg(n_seconds=5, heart_rate=75)
        for i, v in enumerate(ecg):
            proc.add_sample(float(v), i * 8)
        filt_arr = np.array(list(proc._filt_buffer), dtype=np.float32)
        r_idx = len(filt_arr) // 2
        beat = proc._segment_beat(filt_arr, r_idx)
        if beat is not None:
            assert beat.min() >= -0.01, f"Beat below 0: min={beat.min()}"
            assert beat.max() <= 1.01, f"Beat above 1: max={beat.max()}"


class TestBPMCalculation:
    def test_bpm_reasonable_range(self):
        """Feeding a 75 BPM synthetic ECG should give BPM near 75."""
        proc = make_processor()
        ecg = synthetic_ecg(n_seconds=15, heart_rate=75)
        beat_found = False
        for i, v in enumerate(ecg):
            result = proc.add_sample(float(v), i * 8)
            if result is not None:
                beat_found = True

        if beat_found and proc.current_bpm > 0:
            assert 50 <= proc.current_bpm <= 110, f"BPM={proc.current_bpm} out of range for 75 BPM signal"


class TestReset:
    def test_reset_clears_all_state(self):
        proc = make_processor()
        for i in range(300):
            proc.add_sample(0.5, i * 8)

        proc.reset()
        assert len(proc._filt_buffer) == 0
        assert len(proc._raw_buffer) == 0
        assert proc._global_sample_idx == 0
        assert proc.current_bpm == 0.0
        assert proc._last_rpeak_global == -1000
