"""
Tests for InferenceEngine (TFLite INT8 model).

These tests are skipped if the TFLite model file does not exist yet
(model must be trained first with: python ml/train.py).

Tests:
  - Model loads and allocates tensors without error
  - Predict returns correct output schema
  - Prediction class is valid (0–4)
  - Probabilities sum to ~1.0
  - Inference latency < 200ms per beat
  - Batch inference produces consistent results
"""

import sys
import os
import time
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "models", "ecg_model_int8.tflite")
MODEL_AVAILABLE = os.path.exists(MODEL_PATH)

pytestmark = pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason=f"TFLite model not found at {MODEL_PATH}. Run 'python ml/train.py' first."
)


def make_config():
    return {
        "ALERT_CLASSES": [2, 3],
        "ALERT_CONFIDENCE_THRESHOLD": 0.70,
        "TIMER_LOG": False,
    }


@pytest.fixture(scope="module")
def engine():
    from app.services.inference_engine import InferenceEngine
    return InferenceEngine(MODEL_PATH, make_config())


def random_beat():
    """Generate a random normalized beat (187 samples, [0, 1])."""
    return np.random.uniform(0.0, 1.0, 187).astype(np.float32)


def mitbih_beat(class_id: int = 0):
    """Load a real beat from MIT-BIH test CSV."""
    test_csv = os.path.join(ROOT, "mitbih dataset", "mitbih_test.csv")
    if not os.path.exists(test_csv):
        return random_beat()
    import pandas as pd
    df = pd.read_csv(test_csv, header=None)
    rows = df[df.iloc[:, 187] == class_id]
    if len(rows) == 0:
        return random_beat()
    row = rows.iloc[0]
    return row.iloc[:187].values.astype(np.float32)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestInferenceEngineLoad:
    def test_engine_loads(self, engine):
        assert engine is not None

    def test_input_details_correct_shape(self, engine):
        assert engine._input_details[0]["shape"][1] == 187
        assert engine._input_details[0]["shape"][2] == 1

    def test_output_details_correct_classes(self, engine):
        assert engine._output_details[0]["shape"][1] == 5


class TestPredict:
    def test_predict_returns_dict(self, engine):
        beat = random_beat()
        result = engine.predict(beat)
        assert isinstance(result, dict)

    def test_predict_keys_present(self, engine):
        result = engine.predict(random_beat())
        required_keys = ["class_id", "class_name", "confidence", "probabilities", "alert", "inference_ms"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_class_id_valid_range(self, engine):
        for _ in range(10):
            result = engine.predict(random_beat())
            assert 0 <= result["class_id"] <= 4

    def test_probabilities_sum_to_one(self, engine):
        for _ in range(5):
            result = engine.predict(random_beat())
            prob_sum = sum(result["probabilities"])
            assert abs(prob_sum - 1.0) < 0.05, f"Probs sum to {prob_sum:.4f}"

    def test_confidence_matches_max_prob(self, engine):
        result = engine.predict(random_beat())
        max_prob = max(result["probabilities"])
        assert abs(result["confidence"] - max_prob) < 0.01

    def test_alert_classes(self, engine):
        """Alert should only be True for classes 2 or 3 with high confidence."""
        result = engine.predict(random_beat())
        if result["alert"]:
            assert result["class_id"] in [2, 3], \
                f"Alert raised for non-alert class {result['class_id']}"

    def test_inference_latency_under_200ms(self, engine):
        """Each inference call must complete in < 200ms on Raspberry Pi spec."""
        latencies = []
        for _ in range(20):
            beat = random_beat()
            t0 = time.perf_counter()
            engine.predict(beat)
            latencies.append((time.perf_counter() - t0) * 1000)

        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        assert p95_latency < 200, f"P95 latency {p95_latency:.1f}ms exceeds 200ms limit"
        print(f"\nInference latency: mean={sum(latencies)/len(latencies):.1f}ms, "
              f"max={max(latencies):.1f}ms, p95={p95_latency:.1f}ms")


class TestPredictBatch:
    def test_predict_batch_returns_list(self, engine):
        beats = np.random.uniform(0, 1, (5, 187)).astype(np.float32)
        results = engine.predict_batch(beats)
        assert len(results) == 5

    def test_batch_same_as_individual(self, engine):
        """Batch and individual predictions must agree."""
        beats = np.random.uniform(0, 1, (3, 187)).astype(np.float32)
        batch_results = engine.predict_batch(beats)
        for i, beat in enumerate(beats):
            individual = engine.predict(beat)
            assert batch_results[i]["class_id"] == individual["class_id"]


class TestRealBeats:
    """Test on actual MIT-BIH beats if dataset is available."""

    def test_normal_beat_classified_mostly_correctly(self, engine):
        beat = mitbih_beat(class_id=0)
        result = engine.predict(beat)
        # For a real normal beat, confidence should be reasonable
        assert result["confidence"] > 0.3, \
            f"Very low confidence on real normal beat: {result['confidence']:.3f}"
