"""
Custom Keras callbacks for ECG-ResNet-SE training.

Provides:
  - MacroF1Score: batch-accumulated metric monitored by EarlyStopping/LR schedulers
  - ConfusionMatrixCallback: prints normalized CM at epoch end for quick visual feedback
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class MacroF1Score(tf.keras.metrics.Metric):
    """
    Macro-averaged F1 score computed from a running confusion matrix.
    Monitors all 5 classes equally regardless of class frequency —
    critical for detecting degradation on minority classes (S, F).

    Usage:
        model.compile(..., metrics=[MacroF1Score(num_classes=5)])
        # Monitor 'val_macro_f1' in callbacks
    """

    def __init__(self, num_classes: int = 5, name: str = "macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self._cm = self.add_weight(
            name="confusion_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros",
            dtype=tf.float32,
        )

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight=None,
    ) -> None:
        # y_true: one-hot (batch, num_classes) or int (batch,)
        # y_pred: softmax probabilities (batch, num_classes)
        if y_true.shape.rank > 1:
            y_true_int = tf.argmax(y_true, axis=1)
        else:
            y_true_int = tf.cast(y_true, tf.int64)
        y_pred_int = tf.argmax(y_pred, axis=1)

        cm = tf.math.confusion_matrix(
            y_true_int, y_pred_int, num_classes=self.num_classes, dtype=tf.float32
        )
        self._cm.assign_add(cm)

    def result(self) -> tf.Tensor:
        cm = self._cm
        tp = tf.linalg.diag_part(cm)
        fp = tf.reduce_sum(cm, axis=0) - tp
        fn = tf.reduce_sum(cm, axis=1) - tp

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1_per_class = 2.0 * precision * recall / (precision + recall + 1e-7)
        return tf.reduce_mean(f1_per_class)

    def reset_state(self) -> None:
        self._cm.assign(tf.zeros((self.num_classes, self.num_classes)))

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["num_classes"] = self.num_classes
        return cfg


class ConfusionMatrixCallback(Callback):
    """
    Prints a normalized confusion matrix at the end of each epoch.
    Helps catch per-class degradation early during long training runs.
    """

    CLASS_NAMES = ["N", "S", "V", "F", "Q"]

    def __init__(self, val_data: tuple, num_classes: int = 5, print_every: int = 5):
        """
        Args:
            val_data: (X_val, y_val_onehot) numpy arrays.
            num_classes: Number of classes.
            print_every: Print CM every N epochs (reduces log clutter).
        """
        super().__init__()
        self.X_val, self.y_val = val_data
        self.num_classes = num_classes
        self.print_every = print_every

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if (epoch + 1) % self.print_every != 0:
            return

        y_pred_prob = self.model.predict(self.X_val, verbose=0, batch_size=256)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(self.y_val, axis=1)

        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1

        # Normalize row-wise (recall per class)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / np.maximum(row_sums, 1)

        header = "    " + "  ".join(f"{n:>5}" for n in self.CLASS_NAMES)
        print(f"\n[Epoch {epoch + 1}] Normalized Confusion Matrix (rows=True, cols=Pred):")
        print(header)
        for i, row in enumerate(cm_norm):
            row_str = "  ".join(f"{v:5.2f}" for v in row)
            print(f"  {self.CLASS_NAMES[i]} |  {row_str}  | n={row_sums[i][0]}")
        print()


class PerClassF1Callback(Callback):
    """
    Logs per-class F1 scores at the end of each epoch.
    Useful for spotting when the model loses recall on rare classes.
    """

    CLASS_NAMES = ["N (Normal)", "S (SVE)", "V (VE)", "F (Fusion)", "Q (Unknown)"]

    def __init__(self, val_data: tuple, num_classes: int = 5):
        super().__init__()
        self.X_val, self.y_val = val_data
        self.num_classes = num_classes

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        y_pred_prob = self.model.predict(self.X_val, verbose=0, batch_size=256)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(self.y_val, axis=1)

        f1_scores = []
        for c in range(self.num_classes):
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            f1_scores.append(f1)

        print(f"\n[Epoch {epoch + 1}] Per-class F1:")
        for name, score in zip(self.CLASS_NAMES, f1_scores):
            bar = "█" * int(score * 20)
            print(f"  {name:<20} F1={score:.4f}  |{bar:<20}|")
        print(f"  {'Macro F1':<20} F1={np.mean(f1_scores):.4f}")
