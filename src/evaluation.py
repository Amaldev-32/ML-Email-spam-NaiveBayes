"""Model evaluation utilities for SpamShield AI.

All metrics are calculated on the held-out test set only. No training-set
performance is reported as test-set performance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .model import Pipeline


def evaluate_model(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    message_col: str = "message",
    label_col: str = "label",
) -> Dict[str, Any]:
    """Calculate all evaluation metrics on the test set.

    Returns a dictionary with accuracy, precision, recall, F1, confusion matrix,
    classification report, and optionally ROC-AUC.
    """
    X_test = test_df[message_col]
    y_test = test_df[label_col]

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Get class order — ensure "spam" is the positive class for binary metrics
    classes = pipeline.classes_.tolist()
    spam_idx = classes.index("spam") if "spam" in classes else 1
    ham_idx = 1 - spam_idx

    # Binary metrics with spam as positive class
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, pos_label="spam", zero_division=0
    )
    recall = recall_score(y_test, y_pred, pos_label="spam", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="spam", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Classification report as dict
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    )

    # ROC-AUC if probabilities available
    roc_auc = None
    fpr = tpr = thresholds = None
    try:
        y_score = y_proba[:, spam_idx]
        roc_auc = roc_auc_score(y_test.map({"ham": 0, "spam": 1}), y_score)
        fpr, tpr, thresholds = roc_curve(
            y_test.map({"ham": 0, "spam": 1}), y_score
        )
    except Exception:
        pass

    return {
        "accuracy": float(accuracy),
        "precision_spam": float(precision),
        "recall_spam": float(recall),
        "f1_spam": float(f1),
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "classification_report": report_dict,
        "roc_auc": roc_auc,
        "roc_curve": {
            "fpr": fpr.tolist() if fpr is not None else None,
            "tpr": tpr.tolist() if tpr is not None else None,
            "thresholds": thresholds.tolist() if thresholds is not None else None,
        },
        "test_samples": int(len(y_test)),
        "train_samples": None,  # filled by caller if needed
    }


def format_classification_report(report: Dict[str, Any]) -> pd.DataFrame:
    """Convert sklearn classification_report dict to a clean DataFrame for display."""
    rows = []
    for label, metrics in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append(
            {
                "Class": label,
                "Precision": round(metrics.get("precision", 0), 4),
                "Recall": round(metrics.get("recall", 0), 4),
                "F1-Score": round(metrics.get("f1-score", 0), 4),
                "Support": int(metrics.get("support", 0)),
            }
        )
    # Add averages
    if "macro avg" in report:
        rows.append(
            {
                "Class": "macro avg",
                "Precision": round(report["macro avg"]["precision"], 4),
                "Recall": round(report["macro avg"]["recall"], 4),
                "F1-Score": round(report["macro avg"]["f1-score"], 4),
                "Support": int(report["macro avg"]["support"]),
            }
        )
    if "weighted avg" in report:
        rows.append(
            {
                "Class": "weighted avg",
                "Precision": round(report["weighted avg"]["precision"], 4),
                "Recall": round(report["weighted avg"]["recall"], 4),
                "F1-Score": round(report["weighted avg"]["f1-score"], 4),
                "Support": int(report["weighted avg"]["support"]),
            }
        )
    return pd.DataFrame(rows)


def get_confusion_matrix_labels(cm_data: Dict[str, Any]) -> Tuple[List[List[int]], List[str], List[str]]:
    """Return matrix, actual labels, predicted labels for plotting."""
    return (
        cm_data["matrix"],
        ["Ham", "Spam"],
        ["Ham", "Spam"],
    )


def explain_metrics() -> Dict[str, str]:
    """Return educational explanations for each metric."""
    return {
        "accuracy": (
            "Overall fraction of messages classified correctly. "
            "Can be misleading when classes are imbalanced."
        ),
        "precision": (
            "Of all messages the model called spam, how many actually are spam. "
            "High precision means fewer legitimate messages land in the spam folder."
        ),
        "recall": (
            "Of all actual spam messages, how many did the model catch. "
            "High recall means fewer spam messages reach the inbox."
        ),
        "f1_score": (
            "Harmonic mean of precision and recall. Balances both concerns. "
            "Useful single-number summary when you care about both false positives "
            "and false negatives."
        ),
        "true_positives": "Spam messages correctly identified as spam.",
        "true_negatives": "Legitimate messages correctly identified as legitimate.",
        "false_positives": (
            "Legitimate messages incorrectly flagged as spam. "
            "These are the most annoying errors — real messages go to spam folder."
        ),
        "false_negatives": (
            "Spam messages incorrectly passed as legitimate. "
            "These are the dangerous errors — spam reaches the inbox."
        ),
    }