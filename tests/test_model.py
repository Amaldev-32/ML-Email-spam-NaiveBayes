#!/usr/bin/env python3
"""Unit tests for SpamShield AI model module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from src.model import (
    build_pipeline,
    train_model,
    save_model,
    load_model,
    predict_message,
    get_model_info,
)


class TestBuildPipeline:
    """Tests for pipeline construction."""

    def test_pipeline_has_correct_steps(self):
        pipeline = build_pipeline()
        assert "tfidf" in pipeline.named_steps
        assert "clf" in pipeline.named_steps

    def test_pipeline_uses_multinomial_nb(self):
        pipeline = build_pipeline()
        from sklearn.naive_bayes import MultinomialNB
        assert isinstance(pipeline.named_steps["clf"], MultinomialNB)

    def test_pipeline_uses_tfidf(self):
        pipeline = build_pipeline()
        from sklearn.feature_extraction.text import TfidfVectorizer
        assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)

    def test_custom_alpha(self):
        pipeline = build_pipeline(alpha=0.5)
        assert pipeline.named_steps["clf"].alpha == 0.5


class TestTrainModel:
    """Tests for model training."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        return pd.DataFrame({
            "message": [
                "Hello how are you",
                "Free money click here",
                "Meeting at 3pm",
                "Win prize now",
                "Thanks for the update",
                "Urgent: verify account",
            ],
            "label": ["ham", "spam", "ham", "spam", "ham", "spam"],
        })

    def test_train_returns_fitted_pipeline(self, sample_data):
        pipeline = train_model(sample_data)
        # Check that tfidf has vocabulary
        assert hasattr(pipeline.named_steps["tfidf"], "vocabulary_")
        assert len(pipeline.named_steps["tfidf"].vocabulary_) > 0

    def test_train_predicts_correct_classes(self, sample_data):
        pipeline = train_model(sample_data)
        preds = pipeline.predict(sample_data["message"])
        # All predictions should be either ham or spam
        for pred in preds:
            assert pred in ["ham", "spam"]

    def test_train_sets_classes_correctly(self, sample_data):
        pipeline = train_model(sample_data)
        classes = pipeline.classes_.tolist()
        assert set(classes) == {"ham", "spam"}


class TestPredictMessage:
    """Tests for prediction helper."""

    @pytest.fixture
    def trained_pipeline(self):
        df = pd.DataFrame({
            "message": [
                "Hello how are you",
                "Free money click here",
                "Meeting at 3pm",
                "Win prize now",
                "Thanks for the update",
                "Urgent: verify account",
            ],
            "label": ["ham", "spam", "ham", "spam", "ham", "spam"],
        })
        return train_model(df)

    def test_predict_returns_dict(self, trained_pipeline):
        result = predict_message(trained_pipeline, "Hello there")
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, trained_pipeline):
        result = predict_message(trained_pipeline, "Hello there")
        assert "prediction" in result
        assert "spam_probability" in result
        assert "ham_probability" in result
        assert "all_probabilities" in result

    def test_prediction_is_valid_class(self, trained_pipeline):
        result = predict_message(trained_pipeline, "Hello there")
        assert result["prediction"] in ["ham", "spam"]

    def test_probabilities_sum_to_one(self, trained_pipeline):
        result = predict_message(trained_pipeline, "Hello there")
        total = result["spam_probability"] + result["ham_probability"]
        assert abs(total - 1.0) < 1e-6

    def test_probabilities_in_range(self, trained_pipeline):
        result = predict_message(trained_pipeline, "Hello there")
        assert 0 <= result["spam_probability"] <= 1
        assert 0 <= result["ham_probability"] <= 1

    def test_spam_message_high_spam_prob(self, trained_pipeline):
        result = predict_message(trained_pipeline, "Free money click now win prize")
        # Spam-like message should have higher spam probability
        assert result["spam_probability"] > 0.5


class TestGetModelInfo:
    """Tests for model info extraction."""

    @pytest.fixture
    def trained_pipeline(self):
        df = pd.DataFrame({
            "message": ["Hello world", "Free money", "Meeting tomorrow", "Win big"],
            "label": ["ham", "spam", "ham", "spam"],
        })
        return train_model(df)

    def test_info_has_required_keys(self, trained_pipeline):
        info = get_model_info(trained_pipeline)
        assert "model_type" in info
        assert "alpha" in info
        assert "n_features" in info
        assert "classes" in info
        assert "tfidf_params" in info

    def test_model_type_correct(self, trained_pipeline):
        info = get_model_info(trained_pipeline)
        assert info["model_type"] == "MultinomialNB"

    def test_classes_correct(self, trained_pipeline):
        info = get_model_info(trained_pipeline)
        assert set(info["classes"]) == {"ham", "spam"}

    def test_alpha_preserved(self, trained_pipeline):
        info = get_model_info(trained_pipeline)
        assert info["alpha"] == 0.1


class TestSaveLoadModel:
    """Tests for model persistence."""

    @pytest.fixture
    def trained_pipeline(self):
        df = pd.DataFrame({
            "message": ["Hello world", "Free money", "Meeting tomorrow", "Win big"],
            "label": ["ham", "spam", "ham", "spam"],
        })
        return train_model(df)

    def test_save_creates_file(self, trained_pipeline, tmp_path):
        path = tmp_path / "test_model.joblib"
        save_model(trained_pipeline, path)
        assert path.exists()

    def test_load_returns_equivalent_pipeline(self, trained_pipeline, tmp_path):
        path = tmp_path / "test_model.joblib"
        save_model(trained_pipeline, path)
        loaded = load_model(path)
        # Same classes
        assert np.array_equal(loaded.classes_, trained_pipeline.classes_)
        # Same predictions
        test_msgs = ["Hello", "Free money"]
        orig_preds = trained_pipeline.predict(test_msgs)
        loaded_preds = loaded.predict(test_msgs)
        assert np.array_equal(orig_preds, loaded_preds)

    def test_load_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.joblib")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])