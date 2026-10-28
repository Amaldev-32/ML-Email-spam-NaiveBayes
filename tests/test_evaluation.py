#!/usr/bin/env python3
"""Unit tests for SpamShield AI evaluation module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluation import (
    evaluate_model,
    format_classification_report,
    get_confusion_matrix_labels,
    explain_metrics,
)


class TestEvaluateModel:
    """Tests for full evaluation pipeline."""

    @pytest.fixture
    def simple_pipeline(self):
        # Minimal pipeline for reproducible tests
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB(alpha=0.1)),
        ])
        df = pd.DataFrame({
            "message": [
                "Hello how are you",
                "Free money click here now",
                "Meeting at 3pm today",
                "Win prize instantly",
                "Thanks for the update",
                "Urgent verify account quickly",
            ],
            "label": ["ham", "spam", "ham", "spam", "ham", "spam"],
        })
        pipeline.fit(df["message"], df["label"])
        return pipeline

    @pytest.fixture
    def test_df(self):
        return pd.DataFrame({
            "message": [
                "Free prize call now",
                "Meeting at noon",
                "Urgent verify now",
                "Thanks for help",
            ],
            "label": ["spam", "ham", "spam", "ham"],
        })

    def test_evaluate_returns_dict(self, simple_pipeline, test_df):
        result = evaluate_model(simple_pipeline, test_df)
        assert isinstance(result, dict)

    def test_accuracy_in_range(self, simple_pipeline, test_df):
        result = evaluate_model(simple_pipeline, test_df)
        assert 0 <= result["accuracy"] <= 1

    def test_precision_recall_f1_in_range(self, simple_pipeline, test_df):
        result = evaluate_model(simple_pipeline, test_df)
        assert 0 <= result["precision_spam"] <= 1
        assert 0 <= result["recall_spam"] <= 1
        assert 0 <= result["f1_spam"] <= 1

    def test_confusion_matrix_has_all_keys(self, simple_pipeline, test_df):
        result = evaluate_model(simple_pipeline, test_df)
        cm = result["confusion_matrix"]
        assert "matrix" in cm
        assert "true_negatives" in cm
        assert "false_positives" in cm
        assert "false_negatives" in cm
        assert "true_positives" in cm

    def test_confusion_matrix_sums_to_test_samples(self, simple_pipeline, test_df):
        result = evaluate_model(simple_pipeline, test_df)
        cm = result["confusion_matrix"]
        total = cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]
        assert total == len(test_df)

    def test_classification_report_is_dict(self, simple_pipeline, test_df):
        result = evaluate_model(simple_pipeline, test_df)
        report = result["classification_report"]
        assert isinstance(report, dict)
        assert "ham" in report or "spam" in report

    def test_roc_auc_conditional(self, simple_pipeline, test_df):
        result = evaluate_model(simple_pipeline, test_df)
        # ROC-AUC may or may notcompute depending on data
        # We just verify it doesn't crash
        assert "roc_auc" in result

    def test_test_samples_reported(self, simple_pipeline, test_df):
        result = evaluate_model(simple_pipeline, test_df)
        assert result["test_samples"] == len(test_df)


class TestFormatClassificationReport:
    """Tests for report formatting."""

    def test_format_creates_dataframe(self):
        report = {
            "ham": {"precision": 0.95, "recall": 0.90, "f1-score": 0.92, "support": 50},
            "spam": {"precision": 0.88, "recall": 0.85, "f1-score": 0.86, "support": 30},
            "accuracy": 0.89,
            "macro avg": {"precision": 0.92, "recall": 0.88, "f1-score": 0.89, "support": 80},
            "weighted avg": {"precision": 0.92, "recall": 0.89, "f1-score": 0.90, "support": 80},
        }
        df = format_classification_report(report)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 2  # At least ham and spam

    def test_format_includes_averages(self):
        report = {
            "ham": {"precision": 0.8, "recall": 0.8, "f1-score": 0.8, "support": 10},
            "spam": {"precision": 0.7, "recall": 0.7, "f1-score": 0.7, "support": 10},
            "accuracy": 0.75,
            "macro avg": {"precision": 0.75, "recall": 0.75, "f1-score": 0.75, "support": 20},
            "weighted avg": {"precision": 0.75, "recall": 0.75, "f1-score": 0.75, "support": 20},
        }
        df = format_classification_report(report)
        assert "macro avg" in df["Class"].values
        assert "weighted avg" in df["Class"].values


class TestConfusionMatrixLabels:
    """Tests for confusion matrix label helpers."""

    def test_default_labels(self):
        cm = [[10, 2], [1, 8]]
        matrix, actual, predicted = get_confusion_matrix_labels({"matrix": cm})
        assert actual == ["Ham", "Spam"]
        assert predicted == ["Ham", "Spam"]


class TestExplainMetrics:
    """Tests for metric explanations."""

    def test_all_keys_present(self):
        explanations = explain_metrics()
        required = [
            "accuracy", "precision", "recall", "f1_score",
            "true_positives", "true_negatives",
            "false_positives", "false_negatives",
        ]
        for key in required:
            assert key in explanations

    def test_explanations_are_strings(self):
        explanations = explain_metrics()
        for key, value in explanations.items():
            assert isinstance(value, str)
            assert len(value) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])