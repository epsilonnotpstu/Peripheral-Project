"""
Model evaluation and result reporting for ECG-ResNet-SE.

Produces:
  - Classification report (precision, recall, F1, support per class)
  - Normalized confusion matrix
  - Cohen's kappa score
  - Per-class ROC AUC
  - Confusion matrix heatmap (saved as PNG for IEEE paper Figure 5)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


CLASS_NAMES = ["N (Normal)", "S (SVE)", "V (VE)", "F (Fusion)", "Q (Unknown)"]
SHORT_NAMES = ["N", "S", "V", "F", "Q"]


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True,
    save_dir: str = None,
) -> dict:
    """
    Full evaluation of a trained Keras model on the MIT-BIH test set.

    Args:
        model: Loaded Keras model.
        X_test: (N, 187) float32 signal array.
        y_test: (N,) integer labels.
        verbose: Print results to console.
        save_dir: If provided, save confusion matrix PNG and metrics JSON here.

    Returns:
        Dictionary with accuracy, macro_f1, kappa, per_class_metrics, auc_per_class.
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        cohen_kappa_score,
        roc_auc_score,
    )

    X_shaped = X_test.reshape(-1, 187, 1).astype(np.float32)
    y_pred_prob = model.predict(X_shaped, batch_size=256, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # ── Accuracy ───────────────────────────────────────────────────────────────
    accuracy = float(np.mean(y_pred == y_test))

    # ── Classification report ──────────────────────────────────────────────────
    report_str = classification_report(
        y_test, y_pred,
        target_names=SHORT_NAMES,
        digits=4,
    )
    report_dict = classification_report(
        y_test, y_pred,
        target_names=SHORT_NAMES,
        output_dict=True,
    )

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # ── Cohen's Kappa ──────────────────────────────────────────────────────────
    kappa = float(cohen_kappa_score(y_test, y_pred))

    # ── Macro F1 (from report) ─────────────────────────────────────────────────
    macro_f1 = float(report_dict["macro avg"]["f1-score"])
    weighted_f1 = float(report_dict["weighted avg"]["f1-score"])

    # ── Per-class AUC (one-vs-rest) ────────────────────────────────────────────
    from tensorflow.keras.utils import to_categorical
    y_test_oh = to_categorical(y_test, num_classes=5)
    try:
        auc_per_class = roc_auc_score(y_test_oh, y_pred_prob, average=None).tolist()
        macro_auc = float(np.mean(auc_per_class))
    except Exception:
        auc_per_class = [None] * 5
        macro_auc = None

    # ── Print results ──────────────────────────────────────────────────────────
    if verbose:
        sep = "=" * 65
        print(f"\n{sep}")
        print("ECG-ResNet-SE — Evaluation on MIT-BIH Test Set")
        print(sep)
        print(f"\nOverall Accuracy    : {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"Macro F1-Score      : {macro_f1:.4f}")
        print(f"Weighted F1-Score   : {weighted_f1:.4f}")
        print(f"Cohen's Kappa       : {kappa:.4f}")
        if macro_auc:
            print(f"Macro ROC-AUC       : {macro_auc:.4f}")
        print(f"\n{report_str}")

        print("Normalized Confusion Matrix (rows=True class, cols=Predicted):")
        header = "        " + "".join(f"{n:>8}" for n in SHORT_NAMES)
        print(header)
        for i, row in enumerate(cm_norm):
            row_str = "".join(f"{v:8.3f}" for v in row)
            print(f"  {SHORT_NAMES[i]:>5} |{row_str} | n={cm.sum(axis=1)[i]:5,d}")

        if auc_per_class[0] is not None:
            print("\nPer-class ROC AUC (one-vs-rest):")
            for name, auc in zip(SHORT_NAMES, auc_per_class):
                print(f"  {name}: {auc:.4f}")
        print(sep)

    # ── Save figures and metrics ────────────────────────────────────────────────
    results = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
        "macro_auc": macro_auc,
        "auc_per_class": {n: a for n, a in zip(SHORT_NAMES, auc_per_class)},
        "per_class": {
            n: {
                "precision": report_dict[n]["precision"],
                "recall": report_dict[n]["recall"],
                "f1-score": report_dict[n]["f1-score"],
                "support": int(report_dict[n]["support"]),
            }
            for n in SHORT_NAMES
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Save metrics JSON
        metrics_path = os.path.join(save_dir, "eval_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved → {metrics_path}")

        # Save confusion matrix heatmap (IEEE paper Figure 5)
        _plot_confusion_matrix(cm_norm, save_dir)

        # Save ROC curves (IEEE paper Figure 6)
        if auc_per_class[0] is not None:
            _plot_roc_curves(y_test_oh, y_pred_prob, save_dir)

        # Save training-history comparison plot if history exists
        history_path = os.path.join(ROOT, "models", "training_history.json")
        if os.path.exists(history_path):
            _plot_training_history(history_path, save_dir)

    return results


def _plot_confusion_matrix(cm_norm: np.ndarray, save_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=SHORT_NAMES,
        yticklabels=SHORT_NAMES,
        linewidths=0.5,
        ax=ax,
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title("ECG-ResNet-SE: Normalized Confusion Matrix\n(MIT-BIH Test Set)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {path}")


def _plot_roc_curves(y_true_oh: np.ndarray, y_pred_prob: np.ndarray, save_dir: str) -> None:
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#2196F3", "#FF9800", "#F44336", "#9C27B0", "#4CAF50"]

    for i, (name, color) in enumerate(zip(SHORT_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ECG-ResNet-SE: ROC Curves (One-vs-Rest)", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved → {path}")


def _plot_training_history(history_path: str, save_dir: str) -> None:
    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    if "loss" in history:
        axes[0].plot(history["loss"], label="Train Loss", color="#2196F3")
        axes[0].plot(history.get("val_loss", []), label="Val Loss", color="#FF5722")
        axes[0].set_title("Training / Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Macro F1
    if "macro_f1" in history:
        axes[1].plot(history["macro_f1"], label="Train Macro F1", color="#4CAF50")
        axes[1].plot(history.get("val_macro_f1", []), label="Val Macro F1", color="#FF9800")
        axes[1].set_title("Training / Validation Macro F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro F1")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    elif "accuracy" in history:
        axes[1].plot(history["accuracy"], label="Train Acc", color="#4CAF50")
        axes[1].plot(history.get("val_accuracy", []), label="Val Acc", color="#FF9800")
        axes[1].set_title("Training / Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle("ECG-ResNet-SE Training Curves", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved → {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(ROOT, "models", "ecg_resnet_se.keras"))
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "mitbih dataset"))
    parser.add_argument("--save-dir", default=os.path.join(ROOT, "models", "eval_results"))
    args = parser.parse_args()

    import tensorflow as tf
    from ml.data_loader import load_mitbih
    from ml.callbacks import MacroF1Score

    _, _, X_test, y_test = load_mitbih(args.data_dir)
    model = tf.keras.models.load_model(
        args.model, custom_objects={"MacroF1Score": MacroF1Score}
    )
    evaluate_model(model, X_test, y_test, verbose=True, save_dir=args.save_dir)
