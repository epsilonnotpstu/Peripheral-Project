"""
ECG-ResNet-SE Training Script
==============================
Trains the best-possible lightweight 1D CNN model for MIT-BIH 5-class arrhythmia
classification, suitable for TFLite INT8 deployment on Raspberry Pi 3.

Usage:
    python ml/train.py

    # Override config values:
    python ml/train.py --data-dir "mitbih dataset" --epochs 100 --batch-size 64

Outputs:
    models/ecg_resnet_se.keras     — best checkpoint (monitored by val_macro_f1)
    models/ecg_model_int8.tflite   — quantized INT8 model for Raspberry Pi
    models/training_history.json   — training metrics per epoch
"""

import os
import sys
import json
import argparse
import logging
import numpy as np

# ── Path setup so we can import from project root ──────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from ml.model_architecture import build_ecg_resnet_se, compile_model, print_model_summary
from ml.data_loader import load_mitbih, ECGDataSequence, prepare_for_model, to_onehot
from ml.class_balancer import compute_class_weights, apply_smote_tomek, apply_simple_oversample
from ml.callbacks import MacroF1Score, ConfusionMatrixCallback, PerClassF1Callback
from ml.evaluate import evaluate_model
from ml.quantize import quantize_to_int8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECG-ResNet-SE")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to mitbih dataset folder")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-smote", action="store_true", help="Skip SMOTETomek (use simple resample)")
    parser.add_argument("--no-augment", action="store_true", help="Skip on-the-fly augmentation")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--config", type=str, default=os.path.join(ROOT, "config.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})

    # ── Resolve settings (CLI args override config.json) ──────────────────────
    data_dir = args.data_dir or os.path.join(ROOT, train_cfg.get("data_dir", "mitbih dataset"))
    model_dir = os.path.join(ROOT, train_cfg.get("model_output_dir", "models"))
    os.makedirs(model_dir, exist_ok=True)

    epochs = args.epochs or train_cfg.get("epochs", 100)
    batch_size = args.batch_size or train_cfg.get("batch_size", 64)
    dropout_1 = train_cfg.get("dropout_1", 0.4)
    dropout_2 = train_cfg.get("dropout_2", 0.3)
    lr = args.learning_rate
    augment_p = train_cfg.get("augmentation_p", 0.5)

    smote_strategy_raw = train_cfg.get("smote_sampling_strategy", {"0": 20000, "1": 5000, "2": 8000, "3": 3000, "4": 8000})
    smote_strategy = {int(k): int(v) for k, v in smote_strategy_raw.items()}

    keras_path = os.path.join(model_dir, "ecg_resnet_se.keras")
    tflite_path = os.path.join(model_dir, "ecg_model_int8.tflite")
    history_path = os.path.join(model_dir, "training_history.json")

    # ── GPU/CPU setup ──────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        log.info(f"GPUs available: {[g.name for g in gpus]}")
    else:
        log.info("No GPU found — training on CPU (slower but works)")

    tf.random.set_seed(42)
    np.random.seed(42)

    # ── Load dataset ───────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 1/6 — Loading MIT-BIH Dataset")
    log.info("=" * 60)
    X_train_raw, y_train_raw, X_test, y_test = load_mitbih(data_dir)

    # ── Class weights ──────────────────────────────────────────────────────────
    log.info("\nSTEP 2/6 — Computing Class Weights")
    class_weight_dict = compute_class_weights(y_train_raw)

    # ── SMOTETomek resampling ──────────────────────────────────────────────────
    log.info("\nSTEP 3/6 — Resampling Training Data")
    if not args.no_smote:
        try:
            X_train, y_train = apply_smote_tomek(X_train_raw, y_train_raw, smote_strategy)
        except Exception as e:
            log.warning(f"SMOTETomek failed ({e}), falling back to simple oversampling")
            X_train, y_train = apply_simple_oversample(X_train_raw, y_train_raw, smote_strategy)
    else:
        log.info("  Skipping SMOTE (--no-smote flag set)")
        X_train, y_train = apply_simple_oversample(X_train_raw, y_train_raw, smote_strategy)

    log.info(f"  Final training set size: {len(X_train):,} samples")

    # ── Build model ────────────────────────────────────────────────────────────
    log.info("\nSTEP 4/6 — Building ECG-ResNet-SE Model")
    model = build_ecg_resnet_se(
        input_shape=(187, 1),
        num_classes=5,
        dropout_1=dropout_1,
        dropout_2=dropout_2,
    )

    # Add MacroF1Score metric before compiling
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", MacroF1Score(num_classes=5)],
    )
    print_model_summary(model)

    # ── Prepare data sequences ──────────────────────────────────────────────────
    X_test_shaped = prepare_for_model(X_test)
    y_test_onehot = to_onehot(y_test)

    train_gen = ECGDataSequence(
        X_train, y_train,
        batch_size=batch_size,
        augment=not args.no_augment,
        augment_p=augment_p,
        shuffle=True,
    )
    val_gen = ECGDataSequence(
        X_test, y_test,
        batch_size=batch_size,
        augment=False,          # never augment validation
        shuffle=False,
    )

    # ── Callbacks ──────────────────────────────────────────────────────────────
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_macro_f1",
            factor=train_cfg.get("reduce_lr_factor", 0.5),
            patience=train_cfg.get("reduce_lr_patience", 5),
            min_lr=train_cfg.get("min_lr", 1e-6),
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_macro_f1",
            patience=train_cfg.get("early_stopping_patience", 15),
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=keras_path,
            monitor="val_macro_f1",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        ConfusionMatrixCallback(
            val_data=(X_test_shaped, y_test_onehot),
            print_every=5,
        ),
        PerClassF1Callback(
            val_data=(X_test_shaped, y_test_onehot),
        ),
    ]

    # ── Train ──────────────────────────────────────────────────────────────────
    log.info("\nSTEP 5/6 — Training")
    log.info(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    log.info(f"  Monitor: val_macro_f1 (higher = better)")
    log.info(f"  Model checkpoint → {keras_path}")
    log.info("-" * 60)

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    # ── Save training history ──────────────────────────────────────────────────
    history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)
    log.info(f"Training history saved → {history_path}")

    # ── Evaluate on test set ───────────────────────────────────────────────────
    log.info("\nSTEP 6/6 — Final Evaluation on MIT-BIH Test Set")
    log.info("  (Loading best checkpoint for evaluation)")
    best_model = tf.keras.models.load_model(
        keras_path,
        custom_objects={"MacroF1Score": MacroF1Score},
    )
    evaluate_model(best_model, X_test, y_test, verbose=True)

    # ── TFLite INT8 quantization ───────────────────────────────────────────────
    log.info("\nQuantizing to TFLite INT8...")
    representative_data = X_test[:200].reshape(-1, 187, 1).astype(np.float32)
    quantize_to_int8(keras_path, tflite_path, representative_data)
    log.info(f"TFLite model saved → {tflite_path}")

    log.info("\n" + "=" * 60)
    log.info("Training complete!")
    log.info(f"  Keras model : {keras_path}")
    log.info(f"  TFLite model: {tflite_path}")
    log.info(f"  History     : {history_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
