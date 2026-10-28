"""Model training, persistence, and prediction utilities for SpamShield AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .preprocessing import build_tfidf_vectorizer

MODEL_DIR = Path("models")
MODEL_FILENAME = "spam_model.joblib"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

DEFAULT_ALPHA = 0.1


def build_pipeline(alpha: float = DEFAULT_ALPHA) -> Pipeline:
    """Build the full TF-IDF + MultinomialNB pipeline.

    The pipeline is deliberately constructed with sensible defaults but keeps
    the alpha (additive smoothing) parameter configurable so students can
    experiment during their viva.
    """
    return Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer()),
            ("clf", MultinomialNB(alpha=alpha)),
        ]
    )


def train_model(
    train_df: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    message_col: str = "message",
    label_col: str = "label",
) -> Pipeline:
    """Fit the pipeline on the training dataframe only.

    Returns the fitted pipeline, which includes the vectorizer and classifier.
    """
    pipeline = build_pipeline(alpha=alpha)
    pipeline.fit(train_df[message_col], train_df[label_col])
    return pipeline


def save_model(pipeline: Pipeline, path: Optional[Path] = None) -> Path:
    """Persist the fitted pipeline to disk using joblib.

    The path defaults to models/spam_model.joblib inside the project.
    """
    target = path or MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target)
    return target


def load_model(path: Optional[Path] = None) -> Pipeline:
    """Load a previously saved pipeline from disk.

    Raises FileNotFoundError if the artifact is missing or cannot be loaded.
    """
    target = path or MODEL_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {target}. "
            f"Run the training script (scripts/train_model.py) or allow "
            f"the application to train on first launch."
        )
    return joblib.load(target)


def predict_message(pipeline: Pipeline, message: str) -> Dict[str, Any]:
    """Run prediction on a single user message.

    Returns a dictionary with:
    - prediction: "ham" or "spam"
    - spam_probability: float in [0, 1]
    - ham_probability: float in [0, 1]
    - all_probabilities: dict mapping class name to probability
    """
    proba = pipeline.predict_proba([message])[0]
    classes = pipeline.classes_.tolist()
    prob_dict = dict(zip(classes, proba.tolist()))
    predicted_class = pipeline.predict([message])[0]
    return {
        "prediction": predicted_class,
        "spam_probability": prob_dict.get("spam", 0.0),
        "ham_probability": prob_dict.get("ham", 0.0),
        "all_probabilities": prob_dict,
    }


def get_model_info(pipeline: Pipeline) -> Dict[str, Any]:
    """Extract human-readable model metadata for the About/Model Performance pages."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    return {
        "model_type": "MultinomialNB",
        "alpha": clf.alpha,
        "n_features": len(tfidf.vocabulary_) if hasattr(tfidf, "vocabulary_") else None,
        "classes": pipeline.classes_.tolist(),
        "tfidf_params": {
            "max_features": tfidf.max_features,
            "min_df": tfidf.min_df,
            "max_df": tfidf.max_df,
            "ngram_range": tfidf.ngram_range,
        },
    }