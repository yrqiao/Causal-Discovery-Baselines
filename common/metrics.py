"""Evaluation metrics, identical to `calculate_performance_metrics` used in the original notebooks,
plus the Structural Intervention Distance (SID, computed by the R package SID through cdt)."""
import numpy as np


def calculate_performance_metrics(predictions, labels):
    predictions = np.asarray(predictions).astype(bool)
    labels = np.asarray(labels).astype(bool)
    if predictions.shape != labels.shape:
        raise ValueError("Both adjacency matrices must have the same dimensions.")

    TP = np.sum(predictions & labels)
    FP = np.sum(predictions & ~labels)
    FN = np.sum(~predictions & labels)
    TN = np.sum(~predictions & ~labels)

    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    f1_score = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
    # A reversed edge counts as two mistakes (same as cdt.metrics.SHD with double_for_anticausal=True).
    shd = np.sum(predictions != labels)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "f1": float(f1_score),
        "shd": int(shd),
    }


def sid(truth, pred):
    """SID(truth, pred): number of interventional distributions that `pred` gets wrong w.r.t. `truth`."""
    from cdt.metrics import SID
    return int(SID(np.asarray(truth), np.asarray(pred)))


def evaluate(pred, truth):
    metrics = calculate_performance_metrics(pred, truth)
    metrics["sid"] = sid(truth, pred)
    return metrics
