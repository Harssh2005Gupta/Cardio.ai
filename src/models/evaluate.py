"""
src/models/evaluate.py
-------------------------
Full evaluation suite beyond accuracy/ROC-AUC: precision, recall, F1,
PR-AUC, confusion matrix, classification report, ROC curve, PR curve, and
(where meaningful) a calibration curve + Brier score. Saves plots to
reports/ and returns a metrics dict for the model-comparison table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
    }


def classification_report_dict(y_true, y_pred) -> dict:
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def save_evaluation_plots(model_name: str, y_true, y_proba, y_pred, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name}_confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc_score(y_true, y_proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve — {model_name}"); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name}_roc_curve.png", dpi=120)
    plt.close(fig)

    # Precision-recall curve
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(rec, prec, label=f"PR (AUC={average_precision_score(y_true, y_proba):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {model_name}"); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name}_pr_curve.png", dpi=120)
    plt.close(fig)

    # Calibration curve (probability interpretation check)
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(mean_pred, frac_pos, marker="o", label="Model")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration — {model_name}"); ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{model_name}_calibration_curve.png", dpi=120)
        plt.close(fig)
    except Exception:
        pass  # calibration curve can fail on tiny/degenerate test sets; non-critical


def build_comparison_table(all_results: Dict[str, Dict[str, float]]) -> str:
    """Markdown table for the README / CLI output."""
    header = "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |"
    sep = "|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for name, m in all_results.items():
        rows.append(
            f"| {name} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1_score']:.3f} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} |"
        )
    return "\n".join(rows)
