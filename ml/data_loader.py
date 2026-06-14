"""
MIT-BIH Arrhythmia dataset loader with on-the-fly augmentation.

Dataset format (Kaggle preprocessed version):
  - 188 columns: columns 0–186 = normalized ECG signal [0,1], column 187 = class label
  - 5 classes: 0=N, 1=S, 2=V, 3=F, 4=Q
  - Train: mitbih_train.csv (87,554 rows)
  - Test:  mitbih_test.csv  (21,892 rows)
  - Note: folder name has a space — 'mitbih dataset/'
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple


SIGNAL_LEN = 187       # samples per beat window
NUM_CLASSES = 5
LABEL_COL = 187        # 0-indexed column index for class label


def load_mitbih(
    data_dir: str,
    train_file: str = "mitbih_train.csv",
    test_file: str = "mitbih_test.csv",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and validate the MIT-BIH preprocessed dataset.

    Returns:
        X_train: (N_train, 187) float32, values in [0, 1]
        y_train: (N_train,) int32, class labels 0–4
        X_test:  (N_test, 187) float32
        y_test:  (N_test,) int32
    """
    train_path = os.path.join(data_dir, train_file)
    test_path = os.path.join(data_dir, test_file)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    if verbose:
        print(f"Loading train: {train_path}")
    train_df = pd.read_csv(train_path, header=None)

    if verbose:
        print(f"Loading test:  {test_path}")
    test_df = pd.read_csv(test_path, header=None)

    # Validate shape
    assert train_df.shape[1] == 188, f"Expected 188 columns, got {train_df.shape[1]}"
    assert test_df.shape[1] == 188, f"Expected 188 columns, got {test_df.shape[1]}"

    X_train = train_df.iloc[:, :SIGNAL_LEN].values.astype(np.float32)
    y_train = train_df.iloc[:, LABEL_COL].values.astype(np.int32)

    X_test = test_df.iloc[:, :SIGNAL_LEN].values.astype(np.float32)
    y_test = test_df.iloc[:, LABEL_COL].values.astype(np.int32)

    # Clip to [0, 1] — some rows may have tiny float errors
    X_train = np.clip(X_train, 0.0, 1.0)
    X_test = np.clip(X_test, 0.0, 1.0)

    if verbose:
        print(f"\nDataset loaded:")
        print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
        print(f"  X_test : {X_test.shape}   y_test : {y_test.shape}")
        class_names = ["N", "S", "V", "F", "Q"]
        print("\n  Training class distribution:")
        for c in range(NUM_CLASSES):
            count = np.sum(y_train == c)
            pct = count / len(y_train) * 100
            print(f"    {class_names[c]}: {count:6,d}  ({pct:5.1f}%)")

    return X_train, y_train, X_test, y_test


def augment_beat(beat: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Apply random augmentations to a single ECG beat.

    Augmentations (each applied independently with probability p):
      1. Time shift: roll signal by -10 to +10 samples
      2. Amplitude scale: multiply by U(0.8, 1.2), clip to [0, 1]
      3. Gaussian noise: σ=0.02 (conservative — baseline used 0.5, causing instability)
      4. Baseline wander: low-frequency sine wave A∈[0.02,0.05], f∈[0.1,0.5] Hz

    Args:
        beat: (187,) float32 array with values in [0, 1]
        p: Probability of applying each augmentation. Default 0.5.

    Returns:
        Augmented beat, same shape, clipped to [0, 1].
    """
    beat = beat.copy()
    fs = 125.0  # Hz

    # 1. Time shift
    if np.random.rand() < p:
        shift = np.random.randint(-10, 11)
        beat = np.roll(beat, shift)

    # 2. Amplitude scaling
    if np.random.rand() < p:
        scale = np.random.uniform(0.8, 1.2)
        beat = beat * scale

    # 3. Gaussian noise
    if np.random.rand() < p:
        noise = np.random.normal(0.0, 0.02, SIGNAL_LEN).astype(np.float32)
        beat = beat + noise

    # 4. Baseline wander (low-frequency sinusoidal drift)
    if np.random.rand() < p:
        freq = np.random.uniform(0.1, 0.5)
        t = np.arange(SIGNAL_LEN, dtype=np.float32) / fs
        amplitude = np.random.uniform(0.02, 0.05)
        beat = beat + amplitude * np.sin(2.0 * np.pi * freq * t)

    return np.clip(beat, 0.0, 1.0).astype(np.float32)


class ECGDataSequence(tf.keras.utils.Sequence):
    """
    Keras data generator with on-the-fly augmentation.

    Yields batches of shape (batch_size, 187, 1) with one-hot labels (batch_size, 5).
    Augmentation is applied only when training=True.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        num_classes: int = NUM_CLASSES,
        augment: bool = True,
        augment_p: float = 0.5,
        shuffle: bool = True,
    ):
        """
        Args:
            X: (N, 187) signal array
            y: (N,) integer class labels
            batch_size: Samples per batch
            num_classes: For one-hot encoding
            augment: Whether to apply augmentation
            augment_p: Probability per augmentation
            shuffle: Shuffle indices each epoch
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.augment = augment
        self.augment_p = augment_p
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_idx].copy()
        y_batch = self.y[batch_idx]

        if self.augment:
            for i in range(len(X_batch)):
                X_batch[i] = augment_beat(X_batch[i], p=self.augment_p)

        # Reshape to (batch, 187, 1) for Conv1D
        X_batch = X_batch.reshape(-1, SIGNAL_LEN, 1)
        # One-hot encode
        y_onehot = tf.keras.utils.to_categorical(y_batch, num_classes=self.num_classes)

        return X_batch, y_onehot

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


def prepare_for_model(X: np.ndarray) -> np.ndarray:
    """Reshape flat (N, 187) array to (N, 187, 1) for Conv1D input."""
    return X.reshape(-1, SIGNAL_LEN, 1).astype(np.float32)


def to_onehot(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    return tf.keras.utils.to_categorical(y, num_classes=num_classes).astype(np.float32)
