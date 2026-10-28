#!/usr/bin/env python3
"""Training script for SpamShield AI.

This script loads the dataset, trains the TF-IDF + MultinomialNB pipeline,
evaluates it on the held-out test set, and saves the fitted pipeline to disk.

Run from the project root:
    python -m scripts.train_model

The trained model is saved to models/spam_model.joblib and can be loaded by
the Streamlit application for instant predictions without retraining.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is importable when run as a module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data_loader import load_dataset, normalize_labels, safe_drop_bad_rows, inspect_dataset
from src.preprocessing import split_dataset
from src.model import train_model, save_model, get_model_info
from src.evaluation import evaluate_model, format_classification_report


def main() -> int:
    print("=" * 60)
    print("SpamShield AI — Model Training")
    print("=" * 60)

    # 1. Load and inspect raw dataset
    print("\n[1/6] Loading dataset...")
    try:
        raw_df = load_dataset()
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return 1
    print(f"       Loaded {len(raw_df)} raw messages.")

    # 2. Normalize labels
    print("\n[2/6] Normalizing labels...")
    df = normalize_labels(raw_df)

    # 3. Clean data
    print("\n[3/6] Cleaning data...")
    df, clean_meta = safe_drop_bad_rows(df)
    print(f"       Original rows: {clean_meta['original_rows']}")
    print(f"       Rows dropped:  {clean_meta['rows_dropped']}")
    print(f"       Duplicates removed: {clean_meta['duplicates_removed']}")

    # 4. Inspect dataset
    stats = inspect_dataset(df)
    print(f"       Total messages: {stats['total_rows']}")
    print(f"       Ham:            {stats['ham_count']} ({stats['ham_percentage']*100:.1f}%)")
    print(f"       Spam:           {stats['spam_count']} ({stats['spam_percentage']*100:.1f}%)")

    # 5. Train/test split
    print("\n[4/6] Splitting dataset (80/20, stratified)...")
    train_df, test_df = split_dataset(df)
    print(f"       Training:  {len(train_df)} messages")
    print(f"       Test:      {len(test_df)} messages")

    # 6. Train model
    print("\n[5/6] Training TF-IDF + MultinomialNB pipeline...")
    pipeline = train_model(train_df)
    print("       Training complete.")

    # 7. Evaluate on test set
    print("\n[6/6] Evaluating on held-out test set...")
    metrics = evaluate_model(pipeline, test_df)
    metrics["train_samples"] = len(train_df)

    print(f"\n       Test Accuracy:    {metrics['accuracy']:.4f}")
    print(f"       Spam Precision:   {metrics['precision_spam']:.4f}")
    print(f"       Spam Recall:      {metrics['recall_spam']:.4f}")
    print(f"       Spam F1-Score:    {metrics['f1_spam']:.4f}")
    if metrics.get("roc_auc"):
        print(f"       ROC-AUC:          {metrics['roc_auc']:.4f}")

    cm = metrics["confusion_matrix"]
    print(f"\n       Confusion Matrix:")
    print(f"         TN={cm['true_negatives']}  FP={cm['false_positives']}")
    print(f"         FN={cm['false_negatives']}  TP={cm['true_positives']}")

    # 8. Save model
    print("\nSaving trained pipeline...")
    model_path = save_model(pipeline)
    print(f"       Model saved to: {model_path}")

    # Print model info
    info = get_model_info(pipeline)
    print(f"\nModel info:")
    print(f"       Type: {info['model_type']}")
    print(f"       Alpha: {info['alpha']}")
    print(f"       TF-IDF features: {info['n_features']}")
    print(f"       Classes: {info['classes']}")

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())