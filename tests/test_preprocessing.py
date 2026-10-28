#!/usr/bin/env python3
"""Unit tests for SpamShield AI preprocessing module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from src.preprocessing import (
    build_tfidf_vectorizer,
    split_dataset,
    get_default_example_messages,
    preprocess_message,
    validate_message,
    get_feature_names,
)


class TestTfidfVectorizer:
    """Tests for the TF-IDF vectorizer builder."""

    def test_default_parameters(self):
        vec = build_tfidf_vectorizer()
        assert vec.max_features == 10000
        assert vec.min_df == 2
        assert vec.max_df == 0.95
        assert vec.ngram_range == (1, 2)
        assert vec.lowercase is True
        assert vec.strip_accents == "unicode"

    def test_custom_parameters(self):
        vec = build_tfidf_vectorizer(max_features=5000, min_df=1, max_df=0.9, ngram_range=(1, 1))
        assert vec.max_features == 5000
        assert vec.min_df == 1
        assert vec.max_df == 0.9
        assert vec.ngram_range == (1, 1)

    def test_vectorizer_is_unfitted(self):
        vec = build_tfidf_vectorizer()
        assert not hasattr(vec, "vocabulary_") or vec.vocabulary_ is None


class TestSplitDataset:
    """Tests for train/test split."""

    def test_split_produces_two_dataframes(self):
        df = pd.DataFrame({
            "label": ["ham", "spam"] * 50,
            "message": [f"message {i}" for i in range(100)],
        })
        train, test = split_dataset(df)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert len(train) + len(test) == 100

    def test_stratified_split_maintains_class_ratio(self):
        df = pd.DataFrame({
            "label": ["ham"] * 80 + ["spam"] * 20,
            "message": [f"message {i}" for i in range(100)],
        })
        train, test = split_dataset(df, test_size=0.2, random_state=42, stratify=True)
        train_ratio = (train["label"] == "spam").mean()
        test_ratio = (test["label"] == "spam").mean()
        # Both should be close to 0.2
        assert abs(train_ratio - 0.2) < 0.05
        assert abs(test_ratio - 0.2) < 0.05

    def test_non_stratified_split(self):
        df = pd.DataFrame({
            "label": ["ham", "spam"] * 50,
            "message": [f"message {i}" for i in range(100)],
        })
        train, test = split_dataset(df, stratify=False)
        assert len(train) + len(test) == 100


class TestExampleMessages:
    """Tests for example messages."""

    def test_examples_are_dict(self):
        examples = get_default_example_messages()
        assert isinstance(examples, dict)

    def test_examples_not_empty(self):
        examples = get_default_example_messages()
        assert len(examples) >= 4

    def test_examples_are_strings(self):
        examples = get_default_example_messages()
        for key, value in examples.items():
            assert isinstance(value, str)
            assert len(value) > 0


class TestPreprocessMessage:
    """Tests for message preprocessing."""

    def test_strips_whitespace(self):
        assert preprocess_message("  hello  ") == "hello"

    def test_normalizes_multiple_spaces(self):
        assert preprocess_message("hello    world") == "hello world"

    def test_handles_non_string(self):
        assert preprocess_message(123) == ""
        assert preprocess_message(None) == ""

    def test_empty_string(self):
        assert preprocess_message("") == ""

    def test_preserves_urls_and_numbers(self):
        msg = "Call me at 555-1234 or visit http://example.com for $100!"
        result = preprocess_message(msg)
        assert "555-1234" in result
        assert "http://example.com" in result
        assert "$100" in result


class TestValidateMessage:
    """Tests for message validation."""

    def test_valid_message(self):
        valid, error = validate_message("Hello world")
        assert valid is True
        assert error is None

    def test_empty_message(self):
        valid, error = validate_message("")
        assert valid is False
        assert "enter a message" in error.lower()

    def test_whitespace_only(self):
        valid, error = validate_message("   \n\t  ")
        assert valid is False

    def test_too_long(self):
        long_msg = "x" * 10001
        valid, error = validate_message(long_msg)
        assert valid is False
        assert "too long" in error.lower()


class TestFeatureNames:
    """Tests for feature name extraction."""

    def test_unfitted_vectorizer_returns_empty(self):
        vec = build_tfidf_vectorizer()
        names = get_feature_names(vec)
        assert names == []

    def test_fitted_vectorizer_returns_names(self):
        vec = build_tfidf_vectorizer()
        vec.fit(["hello world", "hello spam", "spam ham", "hello world spam", "test message here"])
        names = get_feature_names(vec, top_n=5)
        assert len(names) <= 5  # may return fewer if vocabulary smaller
        assert all(isinstance(n, str) for n in names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])