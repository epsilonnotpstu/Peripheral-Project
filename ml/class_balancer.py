"""
Class imbalance handling for MIT-BIH arrhythmia dataset.

Three complementary strategies used together:
  1. class_weight dict → passed to model.fit() for loss-level weighting
  2. SMOTETomek → synthetic minority oversampling + Tomek link undersampling
  3. ECGDataSequence augmentation (handled in data_loader.py)

Why three strategies?
  - class_weight alone is insufficient when F class has only 641 samples (0.7%)
  - SMOTE alone can introduce unrealistic synthetic beats
  - Together they provide robust training signal for all 5 classes
"""

import numpy as np
from typing import Dict, Tuple


CLASS_NAMES = {0: "N (Normal)", 1: "S (SVE)", 2: "V (VE)", 3: "F (Fusion)", 4: "Q (Unknown)"}


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights using sklearn's formula:
        weight[c] = N_total / (N_classes × N_samples[c])

    Expected output for MIT-BIH train distribution
    (N=72471, S=2223, V=5788, F=641, Q=6431):
        N ≈ 0.24,  S ≈ 7.87,  V ≈ 3.02,  F ≈ 27.24,  Q ≈ 2.72

    Args:
        y: (N,) integer class labels

    Returns:
        Dictionary mapping class index → weight float
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}

    print("\nClass weights computed:")
    for cls_id, weight in class_weight_dict.items():
        count = int(np.sum(y == cls_id))
        print(f"  {CLASS_NAMES.get(cls_id, cls_id):<20}: weight={weight:.4f}  (n={count:,})")

    return class_weight_dict


def apply_smote_tomek(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: Dict[int, int] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTETomek resampling to (X, y).

    SMOTETomek combines:
      - SMOTE: generates synthetic minority samples by interpolating between
               k nearest neighbors in feature space
      - Tomek links: removes borderline majority samples that are too close
                     to minority class centroids

    Args:
        X: (N, 187) signal array (flat, NOT reshaped)
        y: (N,) integer class labels
        sampling_strategy: Dict {class_id: target_count}.
                           Default targets N=20K, S=5K, V=8K, F=3K, Q=8K
        random_state: For reproducibility.
        verbose: Print before/after distribution.

    Returns:
        X_resampled: (M, 187) float32 — M varies based on strategy + Tomek removal
        y_resampled: (M,) int32
    """
    try:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError(
            "imbalanced-learn is required for SMOTE. "
            "Install with: pip install imbalanced-learn"
        )

    if sampling_strategy is None:
        sampling_strategy = {0: 20000, 1: 5000, 2: 8000, 3: 3000, 4: 8000}

    if verbose:
        print("\nBefore SMOTETomek:")
        _print_distribution(y)

    # SMOTETomek requires flat 2D input: (N, features)
    assert X.ndim == 2 and X.shape[1] == 187, f"X must be (N, 187), got {X.shape}"

    smote_tomek = SMOTETomek(
        smote=SMOTE(
            sampling_strategy={k: v for k, v in sampling_strategy.items() if k != 0},
            k_neighbors=5,
            random_state=random_state,
        ),
        random_state=random_state,
    )

    # Handle class 0 (Normal) separately — downsample majority after SMOTE
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

    # Downsample Normal class if it still dominates after Tomek removal
    target_normal = sampling_strategy.get(0, 20000)
    normal_idx = np.where(y_resampled == 0)[0]
    if len(normal_idx) > target_normal:
        rng = np.random.default_rng(random_state)
        keep_normal = rng.choice(normal_idx, size=target_normal, replace=False)
        other_idx = np.where(y_resampled != 0)[0]
        keep_idx = np.concatenate([keep_normal, other_idx])
        rng.shuffle(keep_idx)
        X_resampled = X_resampled[keep_idx]
        y_resampled = y_resampled[keep_idx]

    if verbose:
        print("\nAfter SMOTETomek:")
        _print_distribution(y_resampled)

    return X_resampled.astype(np.float32), y_resampled.astype(np.int32)


def apply_simple_oversample(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: Dict[int, int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback resampling without imblearn: naive random oversampling of minority classes.
    Use this if imbalanced-learn cannot be installed (e.g., resource-constrained Pi).
    """
    if sampling_strategy is None:
        sampling_strategy = {0: 20000, 1: 5000, 2: 8000, 3: 3000, 4: 8000}

    rng = np.random.default_rng(random_state)
    X_out_parts, y_out_parts = [], []

    for cls_id, target in sampling_strategy.items():
        cls_idx = np.where(y == cls_id)[0]
        n_have = len(cls_idx)
        if n_have == 0:
            continue
        chosen = rng.choice(cls_idx, size=target, replace=(n_have < target))
        X_out_parts.append(X[chosen])
        y_out_parts.append(np.full(target, cls_id, dtype=np.int32))

    X_out = np.concatenate(X_out_parts, axis=0)
    y_out = np.concatenate(y_out_parts, axis=0)

    # Shuffle combined result
    perm = rng.permutation(len(X_out))
    return X_out[perm].astype(np.float32), y_out[perm]


def _print_distribution(y: np.ndarray) -> None:
    total = len(y)
    for c in sorted(np.unique(y).astype(int)):
        count = int(np.sum(y == c))
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  Class {c} ({CLASS_NAMES.get(c, '?'):<14}): {count:6,d}  ({pct:5.1f}%)  {bar}")
