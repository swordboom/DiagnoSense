"""
PHASE 5: Evaluation & Reporting
===============================
Evaluates DiagnoSense models and benchmarks against standard baselines
on the same train/val/test splits.
"""

from __future__ import annotations

import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase4_training import DATA_DIR, DEVICE, MODELS_DIR, REPORTS_DIR, HealthModel, MedicineModel, SparseDataset


warnings.filterwarnings("ignore", category=ConvergenceWarning)

METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"
REPORT_MD_PATH = REPORTS_DIR / "evaluation.md"
MEDICINE_THRESHOLD_PATH = MODELS_DIR / "medicine_threshold.json"
SPLIT_METADATA_PATH = REPORTS_DIR / "split_metadata.json"
DATA_QUALITY_PATH = REPORTS_DIR / "data_quality.json"
TRAINING_HISTORY_PATH = REPORTS_DIR / "training_history.json"
PLOTS_DIR = REPORTS_DIR / "plots"
HEALTH_CONFUSION_MATRIX_PATH = PLOTS_DIR / "health_confusion_matrix.png"
ROC_CURVES_PATH = PLOTS_DIR / "roc_auc_curves.png"
PR_CURVES_PATH = PLOTS_DIR / "precision_recall_curves.png"
LOSS_CURVE_PATH = PLOTS_DIR / "training_validation_loss.png"
SYMPTOM_DISEASE_PREFIX = "symptom_disease"


def _predict_health(X_sparse: sp.csr_matrix, model: HealthModel, batch_size: int = 256) -> np.ndarray:
    dummy_y = np.zeros(X_sparse.shape[0], dtype=np.int64)
    loader = DataLoader(SparseDataset(X_sparse, dummy_y, task="multiclass"), batch_size=batch_size, shuffle=False)
    preds: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x.to(DEVICE))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())

    return np.concatenate(preds, axis=0)


def _predict_health_probs(X_sparse: sp.csr_matrix, model: HealthModel, batch_size: int = 256) -> np.ndarray:
    dummy_y = np.zeros(X_sparse.shape[0], dtype=np.int64)
    loader = DataLoader(SparseDataset(X_sparse, dummy_y, task="multiclass"), batch_size=batch_size, shuffle=False)
    probs: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x.to(DEVICE))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.vstack(probs)


def _predict_medicine_probs(X_sparse: sp.csr_matrix, model: MedicineModel, batch_size: int = 256) -> np.ndarray:
    dummy_y = np.zeros((X_sparse.shape[0], 1), dtype=np.float32)
    loader = DataLoader(SparseDataset(X_sparse, dummy_y, task="multilabel"), batch_size=batch_size, shuffle=False)
    probs: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x.to(DEVICE))
            probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.vstack(probs)


def _safe_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _tune_global_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    candidates = np.arange(0.10, 0.91, 0.05)
    best_threshold = 0.50
    best_micro_f1 = -1.0
    scores: Dict[str, float] = {}

    for threshold in candidates:
        y_pred = (y_scores >= threshold).astype(np.int32)
        micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        scores[f"{threshold:.2f}"] = float(micro)
        if micro > best_micro_f1:
            best_micro_f1 = micro
            best_threshold = float(threshold)

    return best_threshold, scores


def _evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def _distribution_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "std": 0.0, "mean": 0.0, "p10": 0.0, "p90": 0.0, "spread": 0.0}
    arr = np.array(values, dtype=np.float32)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "spread": float(np.max(arr) - np.min(arr)),
    }


def _bias_risk_label(spread: float, std: float) -> str:
    if spread > 0.35 or std > 0.15:
        return "high"
    if spread > 0.20 or std > 0.08:
        return "moderate"
    return "low"


def _ensure_plots_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_plot_name(text: str) -> str:
    cleaned = text.replace("_", " ").strip()
    if len(cleaned) <= 25:
        return cleaned
    return cleaned[:22] + "..."


def _safe_metric(metric_fn, *args, **kwargs) -> float:
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return float("nan")


def _safe_roc_auc_multiclass(y_true_bin: np.ndarray, y_scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true_bin, y_scores, average="macro", multi_class="ovr"))
    except ValueError:
        return float("nan")


def _compute_health_curve_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, object]:
    class_ids = np.arange(y_scores.shape[1], dtype=np.int32)
    y_true_bin = label_binarize(y_true, classes=class_ids)
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_scores.ravel())

    metrics = {
        "roc_auc_micro": float(auc(fpr_micro, tpr_micro)),
        "roc_auc_macro": _safe_roc_auc_multiclass(y_true_bin, y_scores),
        "pr_auc_micro": _safe_metric(average_precision_score, y_true_bin, y_scores, average="micro"),
        "pr_auc_macro": _safe_metric(average_precision_score, y_true_bin, y_scores, average="macro"),
    }

    return {
        "metrics": metrics,
        "roc": {"fpr_micro": fpr_micro, "tpr_micro": tpr_micro},
        "pr": {"recall_micro": recall_micro, "precision_micro": precision_micro},
        "positive_prevalence": float(np.mean(y_true_bin)),
    }


def _valid_multilabel_indices(y_true: np.ndarray) -> np.ndarray:
    positives = y_true.sum(axis=0)
    return np.where((positives > 0) & (positives < y_true.shape[0]))[0]


def _compute_medicine_curve_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, object]:
    valid_idx = _valid_multilabel_indices(y_true)
    if valid_idx.size == 0:
        valid_idx = np.arange(y_true.shape[1], dtype=np.int32)

    y_true_valid = y_true[:, valid_idx]
    y_scores_valid = y_scores[:, valid_idx]
    fpr_micro, tpr_micro, _ = roc_curve(y_true_valid.ravel(), y_scores_valid.ravel())
    precision_micro, recall_micro, _ = precision_recall_curve(y_true_valid.ravel(), y_scores_valid.ravel())

    metrics = {
        "roc_auc_micro": float(auc(fpr_micro, tpr_micro)),
        "roc_auc_macro": _safe_metric(roc_auc_score, y_true_valid, y_scores_valid, average="macro"),
        "pr_auc_micro": _safe_metric(average_precision_score, y_true_valid, y_scores_valid, average="micro"),
        "pr_auc_macro": _safe_metric(average_precision_score, y_true_valid, y_scores_valid, average="macro"),
        "labels_used_for_curves": int(valid_idx.size),
    }

    return {
        "metrics": metrics,
        "roc": {"fpr_micro": fpr_micro, "tpr_micro": tpr_micro},
        "pr": {"recall_micro": recall_micro, "precision_micro": precision_micro},
        "positive_prevalence": float(np.mean(y_true_valid)),
    }


def _plot_health_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    path: Path,
    max_classes_to_show: int = 25,
) -> None:
    _ensure_plots_dir()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize="true")

    if len(class_names) > max_classes_to_show:
        support = np.bincount(y_true, minlength=len(class_names))
        top_idx = np.argsort(support)[::-1][:max_classes_to_show]
        cm = cm[np.ix_(top_idx, top_idx)]
        labels = [_sanitize_plot_name(str(class_names[i])) for i in top_idx]
        title_suffix = f" (Top {max_classes_to_show} classes by support)"
    else:
        labels = [_sanitize_plot_name(str(x)) for x in class_names]
        title_suffix = ""

    size = max(10, min(16, 0.45 * len(labels)))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Recall")
    ax.set_title("Health Confusion Matrix (Row-normalized)" + title_suffix)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curves(health_curves: Dict[str, object], medicine_curves: Dict[str, object], path: Path) -> None:
    _ensure_plots_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    h_fpr = health_curves["roc"]["fpr_micro"]
    h_tpr = health_curves["roc"]["tpr_micro"]
    h_metrics = health_curves["metrics"]
    axes[0].plot(h_fpr, h_tpr, linewidth=2.2, label=f"Micro ROC (AUC={h_metrics['roc_auc_micro']:.4f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="gray", label="Random")
    axes[0].set_title("Health ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right")
    axes[0].text(
        0.04,
        0.08,
        f"Macro AUC: {h_metrics['roc_auc_macro']:.4f}",
        transform=axes[0].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    m_fpr = medicine_curves["roc"]["fpr_micro"]
    m_tpr = medicine_curves["roc"]["tpr_micro"]
    m_metrics = medicine_curves["metrics"]
    axes[1].plot(m_fpr, m_tpr, linewidth=2.2, color="#d95f02", label=f"Micro ROC (AUC={m_metrics['roc_auc_micro']:.4f})")
    axes[1].plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="gray", label="Random")
    axes[1].set_title("Medicine ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right")
    axes[1].text(
        0.04,
        0.08,
        f"Macro AUC: {m_metrics['roc_auc_macro']:.4f}",
        transform=axes[1].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.suptitle("ROC Curves (Micro + Macro AUC Summary)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_pr_curves(health_curves: Dict[str, object], medicine_curves: Dict[str, object], path: Path) -> None:
    _ensure_plots_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    h_recall = health_curves["pr"]["recall_micro"]
    h_precision = health_curves["pr"]["precision_micro"]
    h_metrics = health_curves["metrics"]
    h_baseline = health_curves["positive_prevalence"]
    axes[0].plot(h_recall, h_precision, linewidth=2.2, label=f"Micro PR (AP={h_metrics['pr_auc_micro']:.4f})")
    axes[0].hlines(h_baseline, xmin=0, xmax=1, colors="gray", linestyles="--", linewidth=1.2, label="No-skill")
    axes[0].set_title("Health Precision-Recall Curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower left")
    axes[0].text(
        0.04,
        0.08,
        f"Macro AP: {h_metrics['pr_auc_macro']:.4f}",
        transform=axes[0].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    m_recall = medicine_curves["pr"]["recall_micro"]
    m_precision = medicine_curves["pr"]["precision_micro"]
    m_metrics = medicine_curves["metrics"]
    m_baseline = medicine_curves["positive_prevalence"]
    axes[1].plot(m_recall, m_precision, linewidth=2.2, color="#d95f02", label=f"Micro PR (AP={m_metrics['pr_auc_micro']:.4f})")
    axes[1].hlines(m_baseline, xmin=0, xmax=1, colors="gray", linestyles="--", linewidth=1.2, label="No-skill")
    axes[1].set_title("Medicine Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower left")
    axes[1].text(
        0.04,
        0.08,
        f"Macro AP: {m_metrics['pr_auc_macro']:.4f}",
        transform=axes[1].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.suptitle("Precision-Recall Curves (Micro + Macro AP Summary)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_training_vs_validation_loss(path: Path) -> None:
    _ensure_plots_dir()
    history_payload = _load_json_if_exists(TRAINING_HISTORY_PATH)
    health_history = history_payload.get("health", {}).get("history", [])
    if not health_history and "symptom_disease" in history_payload:
        health_history = history_payload.get("symptom_disease", {}).get("history", [])
    medicine_history = history_payload.get("medicine", {}).get("history", [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    if health_history:
        x = [int(row["epoch"]) for row in health_history]
        train = [float(row["train_loss"]) for row in health_history]
        val = [float(row["val_loss"]) for row in health_history]
        axes[0].plot(x, train, label="Train Loss", linewidth=2.0)
        axes[0].plot(x, val, label="Validation Loss", linewidth=2.0)
        axes[0].set_title("Health Loss Curves")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.25)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No health history found", ha="center", va="center")
        axes[0].set_title("Health Loss Curves")
        axes[0].set_axis_off()

    if medicine_history:
        x = [int(row["epoch"]) for row in medicine_history]
        train = [float(row["train_loss"]) for row in medicine_history]
        val = [float(row["val_loss"]) for row in medicine_history]
        axes[1].plot(x, train, label="Train Loss", linewidth=2.0, color="#1b9e77")
        axes[1].plot(x, val, label="Validation Loss", linewidth=2.0, color="#d95f02")
        axes[1].set_title("Medicine Loss Curves")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No medicine history found", ha="center", va="center")
        axes[1].set_title("Medicine Loss Curves")
        axes[1].set_axis_off()

    fig.suptitle("Training vs Validation Loss", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_visualizations(health: Dict[str, object], medicine: Dict[str, object]) -> Dict[str, str]:
    _plot_health_confusion_matrix(
        y_true=np.array(health["y_true"], dtype=np.int32),
        y_pred=np.array(health["y_pred"], dtype=np.int32),
        class_names=[str(x) for x in health["class_names"]],
        path=HEALTH_CONFUSION_MATRIX_PATH,
    )
    _plot_roc_curves(health_curves=health["curve_payload"], medicine_curves=medicine["curve_payload"], path=ROC_CURVES_PATH)
    _plot_pr_curves(health_curves=health["curve_payload"], medicine_curves=medicine["curve_payload"], path=PR_CURVES_PATH)
    _plot_training_vs_validation_loss(path=LOSS_CURVE_PATH)
    return {
        "health_confusion_matrix": str(HEALTH_CONFUSION_MATRIX_PATH.relative_to(REPORTS_DIR).as_posix()),
        "roc_auc_curves": str(ROC_CURVES_PATH.relative_to(REPORTS_DIR).as_posix()),
        "precision_recall_curves": str(PR_CURVES_PATH.relative_to(REPORTS_DIR).as_posix()),
        "training_validation_loss": str(LOSS_CURVE_PATH.relative_to(REPORTS_DIR).as_posix()),
    }


def evaluate_health() -> Dict[str, object]:
    print("\n" + "=" * 70)
    print("  EVALUATING: Health Model")
    print("=" * 70)

    X_train = sp.load_npz(DATA_DIR / f"{SYMPTOM_DISEASE_PREFIX}_X_train.npz")
    y_train = np.load(DATA_DIR / f"{SYMPTOM_DISEASE_PREFIX}_y_train.npy")
    X_test = sp.load_npz(DATA_DIR / f"{SYMPTOM_DISEASE_PREFIX}_X_test.npz")
    y_test = np.load(DATA_DIR / f"{SYMPTOM_DISEASE_PREFIX}_y_test.npy")

    with (MODELS_DIR / "symptom_disease_label_encoder.pkl").open("rb") as handle:
        encoder = pickle.load(handle)
    class_names = list(encoder.classes_)

    model = HealthModel(input_dim=X_test.shape[1], num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(
        torch.load(MODELS_DIR / "symptom_disease_model.pth", map_location=DEVICE, weights_only=True)
    )

    y_pred_train = _predict_health(X_train, model=model)
    y_pred_test = _predict_health(X_test, model=model)
    y_prob_test = _predict_health_probs(X_test, model=model)
    train_metrics = _evaluate_multiclass(y_train, y_pred_train)
    primary = _evaluate_multiclass(y_test, y_pred_test)
    curve_payload = _compute_health_curve_metrics(y_test, y_prob_test)
    curve_metrics = curve_payload["metrics"]

    report = classification_report(
        y_test,
        y_pred_test,
        labels=list(range(len(class_names))),
        target_names=[str(x) for x in class_names],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred_test,
        labels=list(range(len(class_names))),
        target_names=[str(x) for x in class_names],
        zero_division=0,
    )

    print(f"  Accuracy   : {primary['accuracy']:.4f}")
    print(f"  Macro F1   : {primary['macro_f1']:.4f}")
    print(f"  Macro Prec : {primary['macro_precision']:.4f}")
    print(f"  Macro Rec  : {primary['macro_recall']:.4f}")
    print(f"  ROC AUC    : micro={curve_metrics['roc_auc_micro']:.4f} | macro={curve_metrics['roc_auc_macro']:.4f}")
    print(f"  PR AUC     : micro={curve_metrics['pr_auc_micro']:.4f} | macro={curve_metrics['pr_auc_macro']:.4f}")

    train_test_gap_accuracy = float(train_metrics["accuracy"] - primary["accuracy"])
    train_test_gap_macro_f1 = float(train_metrics["macro_f1"] - primary["macro_f1"])
    overfit_flag = bool(train_test_gap_accuracy > 0.02 or train_test_gap_macro_f1 > 0.02)

    recalls = [float(v["recall"]) for k, v in report.items() if isinstance(v, dict) and "support" in v and v["support"] > 0]
    recall_stats = _distribution_stats(recalls)
    lowest_class = min(
        [(k, float(v["recall"])) for k, v in report.items() if isinstance(v, dict) and "support" in v and v["support"] > 0],
        key=lambda x: x[1],
    )
    bias_risk = _bias_risk_label(recall_stats["spread"], recall_stats["std"])

    diagnostics = {
        "train_metrics": train_metrics,
        "train_test_gap": {
            "accuracy": train_test_gap_accuracy,
            "macro_f1": train_test_gap_macro_f1,
        },
        "overfitting_risk": "high" if overfit_flag else "low",
        "recall_distribution": recall_stats,
        "lowest_recall_class": {"class": lowest_class[0], "recall": lowest_class[1]},
        "class_balance_bias_risk": bias_risk,
    }
    print(
        f"  Overfit gap (acc/F1): {train_test_gap_accuracy:.4f}/{train_test_gap_macro_f1:.4f} | "
        f"Bias risk proxy: {bias_risk}"
    )

    return {
        **primary,
        **curve_metrics,
        "test_samples": int(len(y_test)),
        "classification_report": report,
        "classification_report_text": report_text,
        "diagnostics": diagnostics,
        "class_names": [str(x) for x in class_names],
        "y_true": y_test.tolist(),
        "y_pred": y_pred_test.tolist(),
        "curve_payload": curve_payload,
    }


def evaluate_medicine() -> Dict[str, object]:
    print("\n" + "=" * 70)
    print("  EVALUATING: Medicine Side Effects Model")
    print("=" * 70)

    X_train = sp.load_npz(DATA_DIR / "medicine_X_train.npz")
    y_train = np.load(DATA_DIR / "medicine_y_train.npy")
    X_val = sp.load_npz(DATA_DIR / "medicine_X_val.npz")
    y_val = np.load(DATA_DIR / "medicine_y_val.npy")
    X_test = sp.load_npz(DATA_DIR / "medicine_X_test.npz")
    y_test = np.load(DATA_DIR / "medicine_y_test.npy")

    with (MODELS_DIR / "medicine_se_mlb.pkl").open("rb") as handle:
        mlb = pickle.load(handle)
    labels = list(mlb.classes_)

    model = MedicineModel(input_dim=X_test.shape[1], num_labels=len(labels)).to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / "medicine_model.pth", map_location=DEVICE, weights_only=True))

    val_probs = _predict_medicine_probs(X_val, model=model)
    train_probs = _predict_medicine_probs(X_train, model=model)
    test_probs = _predict_medicine_probs(X_test, model=model)
    best_threshold, threshold_scores = _tune_global_threshold(y_val, val_probs)
    y_pred_train = (train_probs >= best_threshold).astype(np.int32)
    y_pred_test = (test_probs >= best_threshold).astype(np.int32)

    train_metrics = _evaluate_multilabel(y_train, y_pred_train)
    primary = _evaluate_multilabel(y_test, y_pred_test)
    curve_payload = _compute_medicine_curve_metrics(y_test, test_probs)
    curve_metrics = curve_payload["metrics"]
    label_report = classification_report(
        y_test,
        y_pred_test,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    with MEDICINE_THRESHOLD_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "global_threshold": best_threshold,
                "fallback_top_k": 5,
                "tuning_metric": "micro_f1_on_validation",
            },
            handle,
            indent=2,
        )

    print(f"  Tuned threshold : {best_threshold:.2f}")
    print(f"  Micro F1        : {primary['micro_f1']:.4f}")
    print(f"  Macro F1        : {primary['macro_f1']:.4f}")
    print(f"  Samples F1      : {primary['samples_f1']:.4f}")
    print(f"  ROC AUC         : micro={curve_metrics['roc_auc_micro']:.4f} | macro={curve_metrics['roc_auc_macro']:.4f}")
    print(f"  PR AUC          : micro={curve_metrics['pr_auc_micro']:.4f} | macro={curve_metrics['pr_auc_macro']:.4f}")

    train_test_gap_micro_f1 = float(train_metrics["micro_f1"] - primary["micro_f1"])
    train_test_gap_samples_f1 = float(train_metrics["samples_f1"] - primary["samples_f1"])
    overfit_flag = bool(train_test_gap_micro_f1 > 0.03 or train_test_gap_samples_f1 > 0.03)

    label_recalls = [float(v["recall"]) for k, v in label_report.items() if isinstance(v, dict) and "support" in v and v["support"] > 0]
    recall_stats = _distribution_stats(label_recalls)
    bias_risk = _bias_risk_label(recall_stats["spread"], recall_stats["std"])

    diagnostics = {
        "train_metrics": train_metrics,
        "train_test_gap": {
            "micro_f1": train_test_gap_micro_f1,
            "samples_f1": train_test_gap_samples_f1,
        },
        "overfitting_risk": "high" if overfit_flag else "low",
        "label_recall_distribution": recall_stats,
        "label_balance_bias_risk": bias_risk,
    }
    print(
        f"  Overfit gap (micro/samples F1): {train_test_gap_micro_f1:.4f}/{train_test_gap_samples_f1:.4f} | "
        f"Bias risk proxy: {bias_risk}"
    )

    return {
        **primary,
        **curve_metrics,
        "test_samples": int(len(y_test)),
        "num_labels": int(len(labels)),
        "threshold": float(best_threshold),
        "threshold_scores": threshold_scores,
        "classification_report": label_report,
        "diagnostics": diagnostics,
        "curve_payload": curve_payload,
    }


def benchmark_health_models() -> List[Dict[str, object]]:
    print("\n" + "=" * 70)
    print("  BENCHMARK: Health Baselines")
    print("=" * 70)

    X_train = sp.load_npz(DATA_DIR / f"{SYMPTOM_DISEASE_PREFIX}_X_train.npz")
    y_train = np.load(DATA_DIR / f"{SYMPTOM_DISEASE_PREFIX}_y_train.npy")
    X_test = sp.load_npz(DATA_DIR / f"{SYMPTOM_DISEASE_PREFIX}_X_test.npz")
    y_test = np.load(DATA_DIR / f"{SYMPTOM_DISEASE_PREFIX}_y_test.npy")

    baselines = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent", random_state=42)),
        ("MultinomialNB", MultinomialNB(alpha=0.1)),
        ("LogisticRegression", LogisticRegression(max_iter=500, solver="saga", n_jobs=1, random_state=42)),
        ("LinearSVC", LinearSVC(random_state=42)),
    ]

    results: List[Dict[str, object]] = []
    for name, model in baselines:
        start = time.perf_counter()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.perf_counter() - start
        metrics = _evaluate_multiclass(y_test, y_pred)
        row = {"model": name, **metrics, "fit_eval_seconds": float(elapsed)}
        results.append(row)
        print(f"  {name:<20} Macro-F1={row['macro_f1']:.4f} Accuracy={row['accuracy']:.4f}")

    return sorted(results, key=lambda x: x["macro_f1"], reverse=True)


def _to_prob_scores(model: OneVsRestClassifier, X: sp.csr_matrix) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=np.float32)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=np.float32)
        return _safe_sigmoid(raw)
    pred = np.asarray(model.predict(X), dtype=np.float32)
    return pred


def benchmark_medicine_models() -> List[Dict[str, object]]:
    print("\n" + "=" * 70)
    print("  BENCHMARK: Medicine Baselines")
    print("=" * 70)

    X_train = sp.load_npz(DATA_DIR / "medicine_X_train.npz")
    y_train = np.load(DATA_DIR / "medicine_y_train.npy")
    X_val = sp.load_npz(DATA_DIR / "medicine_X_val.npz")
    y_val = np.load(DATA_DIR / "medicine_y_val.npy")
    X_test = sp.load_npz(DATA_DIR / "medicine_X_test.npz")
    y_test = np.load(DATA_DIR / "medicine_y_test.npy")

    baselines = [
        ("OVR-MultinomialNB", OneVsRestClassifier(MultinomialNB(alpha=0.2), n_jobs=1)),
        (
            "OVR-SGDLogLoss",
            OneVsRestClassifier(
                SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=35, tol=1e-3, random_state=42),
                n_jobs=1,
            ),
        ),
    ]

    results: List[Dict[str, object]] = []
    for name, model in baselines:
        start = time.perf_counter()
        model.fit(X_train, y_train)
        val_scores = _to_prob_scores(model, X_val)
        test_scores = _to_prob_scores(model, X_test)

        threshold, _ = _tune_global_threshold(y_val, val_scores)
        y_pred_test = (test_scores >= threshold).astype(np.int32)
        elapsed = time.perf_counter() - start

        metrics = _evaluate_multilabel(y_test, y_pred_test)
        row = {
            "model": name,
            **metrics,
            "threshold": float(threshold),
            "fit_eval_seconds": float(elapsed),
        }
        results.append(row)
        print(f"  {name:<20} Micro-F1={row['micro_f1']:.4f} Samples-F1={row['samples_f1']:.4f}")

    return sorted(results, key=lambda x: x["micro_f1"], reverse=True)


def _load_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _build_markdown_table(rows: List[Dict[str, object]], columns: List[str], percentage_cols: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: List[str] = []

    for row in rows:
        rendered = []
        for col in columns:
            value = row.get(col, "")
            if col in percentage_cols and isinstance(value, (float, int)):
                rendered.append(_format_pct(float(value)))
            elif isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        body.append("| " + " | ".join(rendered) + " |")

    return "\n".join([header, separator, *body])


def _primary_metric(name: str, value: float, rationale: str) -> Dict[str, object]:
    return {
        "name": str(name),
        "value": float(value),
        "rationale": str(rationale),
    }


def write_reports(
    health: Dict[str, object],
    medicine: Dict[str, object],
    health_baselines: List[Dict[str, object]],
    medicine_baselines: List[Dict[str, object]],
    plots: Dict[str, str],
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    split_meta = _load_json_if_exists(SPLIT_METADATA_PATH)
    quality_meta = _load_json_if_exists(DATA_QUALITY_PATH)

    health_comparison = sorted(
        [
            {
                "model": "DiagnoSense-MLP",
                **{k: health[k] for k in ("accuracy", "micro_f1", "macro_f1", "macro_precision", "macro_recall")},
            }
        ]
        + health_baselines,
        key=lambda row: float(row["macro_f1"]),
        reverse=True,
    )

    medicine_comparison = sorted(
        [
            {
                "model": "DiagnoSense-MLP",
                **{
                    k: medicine[k]
                    for k in (
                        "subset_accuracy",
                        "micro_f1",
                        "macro_f1",
                        "samples_f1",
                        "micro_precision",
                        "micro_recall",
                    )
                },
                "threshold": medicine["threshold"],
            }
        ]
        + medicine_baselines,
        key=lambda row: float(row["micro_f1"]),
        reverse=True,
    )

    strong_health_baselines = [row for row in health_comparison if row["model"] != "DummyMostFrequent"]
    health_overseparable_warning = bool(
        len(strong_health_baselines) >= 2
        and health["macro_f1"] >= 0.995
        and min(float(row["macro_f1"]) for row in strong_health_baselines[:3]) >= 0.995
    )
    health["diagnostics"]["dataset_realism_warning"] = health_overseparable_warning

    metrics_payload = {
        "health": {
            "task_type": "single_label_multiclass",
            "primary_metric": _primary_metric(
                name="accuracy",
                value=health["accuracy"],
                rationale="Single-label multiclass task with exactly one disease prediction per sample.",
            ),
            "test_accuracy": health["accuracy"],
            "micro_f1": health["micro_f1"],
            "macro_f1": health["macro_f1"],
            "macro_precision": health["macro_precision"],
            "macro_recall": health["macro_recall"],
            "roc_auc_micro": health["roc_auc_micro"],
            "roc_auc_macro": health["roc_auc_macro"],
            "pr_auc_micro": health["pr_auc_micro"],
            "pr_auc_macro": health["pr_auc_macro"],
            "test_samples": health["test_samples"],
            "diagnostics": health["diagnostics"],
        },
        "medicine": {
            "task_type": "multilabel",
            "primary_metric": _primary_metric(
                name="micro_f1",
                value=medicine["micro_f1"],
                rationale="Multi-label prediction with class imbalance; micro-F1 balances precision and recall over all labels.",
            ),
            "subset_accuracy": medicine["subset_accuracy"],
            "micro_f1": medicine["micro_f1"],
            "macro_f1": medicine["macro_f1"],
            "samples_f1": medicine["samples_f1"],
            "micro_precision": medicine["micro_precision"],
            "micro_recall": medicine["micro_recall"],
            "roc_auc_micro": medicine["roc_auc_micro"],
            "roc_auc_macro": medicine["roc_auc_macro"],
            "pr_auc_micro": medicine["pr_auc_micro"],
            "pr_auc_macro": medicine["pr_auc_macro"],
            "test_samples": medicine["test_samples"],
            "num_labels": medicine["num_labels"],
            "threshold": medicine["threshold"],
            "labels_used_for_curves": medicine["labels_used_for_curves"],
            "diagnostics": medicine["diagnostics"],
        },
        "comparison": {
            "health": health_comparison,
            "medicine": medicine_comparison,
        },
        "plots": plots,
        "split_metadata": split_meta,
        "data_quality": quality_meta,
    }

    with METRICS_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    health_table = _build_markdown_table(
        rows=health_comparison,
        columns=["model", "accuracy", "macro_f1", "macro_precision", "macro_recall"],
        percentage_cols=["accuracy", "macro_f1", "macro_precision", "macro_recall"],
    )
    medicine_table = _build_markdown_table(
        rows=medicine_comparison,
        columns=[
            "model",
            "subset_accuracy",
            "micro_f1",
            "macro_f1",
            "samples_f1",
            "micro_precision",
            "micro_recall",
            "threshold",
        ],
        percentage_cols=["subset_accuracy", "micro_f1", "macro_f1", "samples_f1", "micro_precision", "micro_recall"],
    )

    content = f"""# DiagnoSense Evaluation Report

## Executive Summary
This report summarizes the final evaluation for the two DiagnoSense pipelines after leakage-safe cleaning, deduplication, and group-aware splitting.

- Health pipeline accuracy: **{health['accuracy']:.2%}**
- Health pipeline macro-F1: **{health['macro_f1']:.2%}**
- Health ROC-AUC (micro/macro): **{health['roc_auc_micro']:.4f} / {health['roc_auc_macro']:.4f}**
- Medicine pipeline micro-F1: **{medicine['micro_f1']:.2%}**
- Medicine pipeline samples-F1: **{medicine['samples_f1']:.2%}**
- Medicine ROC-AUC (micro/macro): **{medicine['roc_auc_micro']:.4f} / {medicine['roc_auc_macro']:.4f}**
- Tuned global medicine threshold: **{medicine['threshold']:.2f}**

## Primary Metric Selection
- Health primary metric: **Accuracy** (single-label multiclass classification with one ground-truth disease per sample).
- Medicine primary metric: **Micro-F1** (multi-label prediction where each sample can have multiple side effects and labels are imbalanced).
- Cross-task references: health micro-F1 = **{health['micro_f1']:.2%}**, medicine subset accuracy (exact-match) = **{medicine['subset_accuracy']:.2%}**.

## Health Pipeline (Symptom -> Disease)
- Test samples: {health['test_samples']}
- Accuracy: {health['accuracy']:.4f}
- Micro F1 (reference): {health['micro_f1']:.4f}
- Macro precision: {health['macro_precision']:.4f}
- Macro recall: {health['macro_recall']:.4f}
- Macro F1: {health['macro_f1']:.4f}
- ROC-AUC (micro / macro): {health['roc_auc_micro']:.4f} / {health['roc_auc_macro']:.4f}
- PR-AUC (micro / macro): {health['pr_auc_micro']:.4f} / {health['pr_auc_macro']:.4f}

### Classification Report
```text
{health['classification_report_text']}
```

## Medicine Pipeline (Context -> Side Effects)
- Test samples: {medicine['test_samples']}
- Labels evaluated: {medicine['num_labels']}
- Threshold tuning metric: validation micro-F1
- Selected threshold: {medicine['threshold']:.2f}
- Subset accuracy (exact match, reference): {medicine['subset_accuracy']:.4f}
- Micro precision: {medicine['micro_precision']:.4f}
- Micro recall: {medicine['micro_recall']:.4f}
- Micro F1: {medicine['micro_f1']:.4f}
- Macro F1: {medicine['macro_f1']:.4f}
- Samples F1: {medicine['samples_f1']:.4f}
- ROC-AUC (micro / macro): {medicine['roc_auc_micro']:.4f} / {medicine['roc_auc_macro']:.4f}
- PR-AUC (micro / macro): {medicine['pr_auc_micro']:.4f} / {medicine['pr_auc_macro']:.4f}
- Labels used for curve metrics: {medicine['labels_used_for_curves']}

## Overfitting & Bias Diagnostics

### Health Diagnostics
- Train accuracy: {health['diagnostics']['train_metrics']['accuracy']:.4f}
- Train macro-F1: {health['diagnostics']['train_metrics']['macro_f1']:.4f}
- Train-test accuracy gap: {health['diagnostics']['train_test_gap']['accuracy']:.4f}
- Train-test macro-F1 gap: {health['diagnostics']['train_test_gap']['macro_f1']:.4f}
- Overfitting risk: **{health['diagnostics']['overfitting_risk']}**
- Class recall spread: {health['diagnostics']['recall_distribution']['spread']:.4f}
- Class balance bias risk proxy: **{health['diagnostics']['class_balance_bias_risk']}**
- Lowest recall class: {health['diagnostics']['lowest_recall_class']['class']} ({health['diagnostics']['lowest_recall_class']['recall']:.4f})
- Dataset realism warning (over-separable): **{str(health['diagnostics'].get('dataset_realism_warning', False)).lower()}**

### Medicine Diagnostics
- Train micro-F1: {medicine['diagnostics']['train_metrics']['micro_f1']:.4f}
- Train samples-F1: {medicine['diagnostics']['train_metrics']['samples_f1']:.4f}
- Train-test micro-F1 gap: {medicine['diagnostics']['train_test_gap']['micro_f1']:.4f}
- Train-test samples-F1 gap: {medicine['diagnostics']['train_test_gap']['samples_f1']:.4f}
- Overfitting risk: **{medicine['diagnostics']['overfitting_risk']}**
- Label recall spread: {medicine['diagnostics']['label_recall_distribution']['spread']:.4f}
- Label balance bias risk proxy: **{medicine['diagnostics']['label_balance_bias_risk']}**

## Standard Model Benchmarks (Same Splits)

### Health (Multi-Class)
{health_table}

### Medicine (Multi-Label)
{medicine_table}

## Visualization Artifacts
- Health confusion matrix: `./{plots['health_confusion_matrix']}`
- ROC curve + AUC: `./{plots['roc_auc_curves']}`
- Precision-recall curve: `./{plots['precision_recall_curves']}`
- Training vs validation loss: `./{plots['training_validation_loss']}`

## Notes
- Split metadata and leakage checks: `reports/split_metadata.json`
- Data quality summary: `reports/data_quality.json`
- Machine-readable metrics: `reports/metrics.json`
"""

    with REPORT_MD_PATH.open("w", encoding="utf-8") as handle:
        handle.write(content)

    print(f"  Saved metrics JSON: {METRICS_JSON_PATH}")
    print(f"  Saved report: {REPORT_MD_PATH}")


if __name__ == "__main__":
    health_metrics = evaluate_health()
    medicine_metrics = evaluate_medicine()
    plot_artifacts = generate_visualizations(health_metrics, medicine_metrics)
    health_baseline_results = benchmark_health_models()
    medicine_baseline_results = benchmark_medicine_models()
    write_reports(
        health_metrics,
        medicine_metrics,
        health_baseline_results,
        medicine_baseline_results,
        plots=plot_artifacts,
    )
    print("\n  [DONE] PHASE 5 COMPLETE")
