"""
ECG-ResNet-SE: Multi-scale Residual 1D CNN with Squeeze-and-Excitation Attention.

Architecture designed for:
- MIT-BIH 5-class arrhythmia classification (N, S, V, F, Q)
- Input: (187, 1) — normalized ECG beat windows at 125 Hz
- Target: >97% accuracy, >0.95 macro F1 on MIT-BIH test set
- Inference: <40ms per beat on Raspberry Pi 3 (ARM Cortex-A53 @ 1.2GHz)
- Model size: ~250K parameters → ~250 KB INT8 TFLite
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input


def _se_block(x: tf.Tensor, ratio: int = 4) -> tf.Tensor:
    """
    Squeeze-and-Excitation block for 1D feature maps.
    Learns per-channel importance weights — critical for distinguishing
    subtle morphological differences between S, V, and F beat classes.
    """
    filters = x.shape[-1]
    squeeze = layers.GlobalAveragePooling1D()(x)             # (batch, filters)
    excite = layers.Dense(max(1, filters // ratio), activation="relu")(squeeze)
    excite = layers.Dense(filters, activation="sigmoid")(excite)
    excite = layers.Reshape((1, filters))(excite)            # (batch, 1, filters)
    return layers.Multiply()([x, excite])


def _residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    se_ratio: int = 4,
    downsample: bool = False,
) -> tf.Tensor:
    """
    Pre-activation residual block with SE attention.
    Uses identity shortcut when filter dimensions match; 1×1 projection otherwise.
    """
    shortcut = x

    # Main path
    h = layers.Conv1D(filters, kernel_size, padding="same", use_bias=False)(x)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)
    h = layers.Conv1D(filters, kernel_size, padding="same", use_bias=False)(h)
    h = layers.BatchNormalization()(h)

    # Squeeze-and-Excitation on main path
    h = _se_block(h, ratio=se_ratio)

    # Projection shortcut if filter count changes
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    return layers.Activation("relu")(layers.Add()([h, shortcut]))


def build_ecg_resnet_se(
    input_shape: tuple = (187, 1),
    num_classes: int = 5,
    dropout_1: float = 0.4,
    dropout_2: float = 0.3,
) -> Model:
    """
    Build the ECG-ResNet-SE model.

    Architecture overview:
        Stem (32 filters, k=7)
        → Stage 1: ResBlock(32, k=5) × 2 + SE → MaxPool(2)   → 93 steps
        → Stage 2: ResBlock(64, k=5) × 2 + SE → MaxPool(2)   → 46 steps
        → Stage 3: ResBlock(128, k=3) × 2 + SE → MaxPool(2)  → 23 steps
        → GlobalAvgPool ⊕ GlobalMaxPool → (256,)
        → Dense(128) + BN + ReLU + Dropout(dropout_1)
        → Dense(64) + BN + ReLU + Dropout(dropout_2)
        → Dense(num_classes, softmax)

    Args:
        input_shape: (signal_length, channels). Default (187, 1).
        num_classes: Number of arrhythmia classes. Default 5.
        dropout_1: Dropout rate after first dense layer.
        dropout_2: Dropout rate after second dense layer.

    Returns:
        Uncompiled Keras Model.
    """
    inputs = Input(shape=input_shape, name="ecg_input")

    # ── Stem ────────────────────────────────────────────────────────────────
    x = layers.Conv1D(32, 7, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    # ── Stage 1: 32 filters, kernel=5 ───────────────────────────────────────
    x = _residual_block(x, filters=32, kernel_size=5, se_ratio=4)
    x = _residual_block(x, filters=32, kernel_size=5, se_ratio=4)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)          # → (93, 32)

    # ── Stage 2: 64 filters, kernel=5 ───────────────────────────────────────
    x = _residual_block(x, filters=64, kernel_size=5, se_ratio=4)
    x = _residual_block(x, filters=64, kernel_size=5, se_ratio=4)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)          # → (46, 64)

    # ── Stage 3: 128 filters, kernel=3 ──────────────────────────────────────
    x = _residual_block(x, filters=128, kernel_size=3, se_ratio=8)
    x = _residual_block(x, filters=128, kernel_size=3, se_ratio=8)
    x = layers.MaxPooling1D(pool_size=2, name="pool3")(x)          # → (23, 128)

    # ── Dual-pooling aggregation head ───────────────────────────────────────
    # GlobalAvgPool captures smooth signal context; GlobalMaxPool sharpens on R-peaks
    avg_pool = layers.GlobalAveragePooling1D(name="gap")(x)
    max_pool = layers.GlobalMaxPooling1D(name="gmp")(x)
    x = layers.Concatenate(name="dual_pool")([avg_pool, max_pool])  # → (256,)

    # ── Classifier ──────────────────────────────────────────────────────────
    x = layers.Dense(128, use_bias=False, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_relu")(x)
    x = layers.Dropout(dropout_1, name="drop1")(x)

    x = layers.Dense(64, use_bias=False, name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Activation("relu", name="fc2_relu")(x)
    x = layers.Dropout(dropout_2, name="drop2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="ECG_ResNet_SE")
    return model


def compile_model(model: Model, learning_rate: float = 1e-3) -> Model:
    """Compile with Adam + categorical crossentropy + label smoothing."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


def print_model_summary(model: Model) -> None:
    model.summary()
    total_params = model.count_params()
    print(f"\nTotal parameters : {total_params:,}")
    print(f"Float32 size     : {total_params * 4 / 1024:.1f} KB")
    print(f"INT8 est. size   : {total_params / 1024:.1f} KB")


if __name__ == "__main__":
    model = build_ecg_resnet_se()
    compile_model(model)
    print_model_summary(model)
