"""
TFLite inference engine for per-beat arrhythmia classification.

Handles INT8 quantization transparently:
  - float32 input beat → quantize to int8 → TFLite invoke → dequantize output → float32 probs

Supports both tflite_runtime (Raspberry Pi, ~3 MB) and tensorflow.lite
(development machine, full TensorFlow). Falls back gracefully.
"""

import time
import logging
import numpy as np
import json
import os

log = logging.getLogger(__name__)


CLASS_NAMES = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown",
}
SHORT_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}


class InferenceEngine:
    """
    Runs arrhythmia classification on 187-sample beat windows using a
    TFLite INT8 quantized model.

    Input:  np.ndarray shape (187,) with values normalized to [0, 1]
    Output: dict with class_id, class_name, confidence, probabilities, alert
    """

    def __init__(self, model_path: str, app_config: dict):
        self._model_path = model_path
        self._alert_classes = set(app_config.get("ALERT_CLASSES", [2, 3]))
        self._alert_threshold = float(app_config.get("ALERT_CONFIDENCE_THRESHOLD", 0.70))
        self._timer_log = app_config.get("TIMER_LOG", False)

        # Load interpreter (tflite_runtime preferred over full TF on Pi)
        self._interpreter = self._load_interpreter(model_path)
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Quantization parameters for INT8 ↔ float32 conversion
        self._input_scale = float(self._input_details[0]["quantization"][0])
        self._input_zero_pt = int(self._input_details[0]["quantization"][1])
        self._output_scale = float(self._output_details[0]["quantization"][0])
        self._output_zero_pt = int(self._output_details[0]["quantization"][1])

        self._is_quantized = self._input_details[0]["dtype"] == np.int8

        log.info(
            f"InferenceEngine loaded: {os.path.basename(model_path)}, "
            f"quantized={self._is_quantized}, "
            f"input_scale={self._input_scale:.6f}"
        )

    @staticmethod
    def _load_interpreter(model_path: str):
        """Load TFLite interpreter — prefer tflite_runtime on Pi for fast startup."""
        try:
            import tflite_runtime.interpreter as tflite
            return tflite.Interpreter(model_path=model_path)
        except ImportError:
            pass
        try:
            import tensorflow.lite as tflite
            return tflite.Interpreter(model_path=model_path)
        except ImportError:
            raise ImportError(
                "Neither tflite_runtime nor tensorflow is installed. "
                "On Raspberry Pi: pip install tflite-runtime\n"
                "On dev machine: pip install tensorflow"
            )

    def predict(self, beat: np.ndarray) -> dict:
        """
        Classify a single ECG beat.

        Args:
            beat: np.ndarray shape (187,), values in [0, 1] (float32)

        Returns:
            {
                class_id: int (0–4),
                class_name: str,
                short_name: str,
                confidence: float (0–1),
                probabilities: list[float] (5 values),
                alert: bool,
                inference_ms: float
            }
        """
        t0 = time.perf_counter()

        # Reshape to (1, 187, 1)
        input_data = beat.reshape(1, 187, 1).astype(np.float32)

        if self._is_quantized:
            # float32 → int8
            if self._input_scale > 0:
                input_int8 = (input_data / self._input_scale + self._input_zero_pt).astype(np.int8)
            else:
                input_int8 = input_data.astype(np.int8)
            self._interpreter.set_tensor(self._input_details[0]["index"], input_int8)
        else:
            self._interpreter.set_tensor(self._input_details[0]["index"], input_data)

        self._interpreter.invoke()

        if self._is_quantized:
            output_int8 = self._interpreter.get_tensor(self._output_details[0]["index"])
            # int8 → float32
            output_float = (
                (output_int8.astype(np.float32) - self._output_zero_pt) * self._output_scale
            )
        else:
            output_float = self._interpreter.get_tensor(self._output_details[0]["index"])

        probabilities = output_float[0].tolist()

        # Ensure probabilities are non-negative and sum to ~1
        probabilities = [max(0.0, p) for p in probabilities]
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]

        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])

        inference_ms = (time.perf_counter() - t0) * 1000.0

        if self._timer_log:
            log.debug(f"Inference: {inference_ms:.1f}ms → class={class_id} ({confidence:.3f})")

        return {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "short_name": SHORT_NAMES[class_id],
            "confidence": round(confidence, 4),
            "probabilities": [round(p, 4) for p in probabilities],
            "alert": class_id in self._alert_classes and confidence >= self._alert_threshold,
            "inference_ms": round(inference_ms, 2),
        }

    def predict_batch(self, beats: np.ndarray) -> list:
        """Classify multiple beats. beats: (N, 187) array. Returns list of dicts."""
        return [self.predict(beat) for beat in beats]
