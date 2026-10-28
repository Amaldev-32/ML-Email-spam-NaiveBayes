"""Text preprocessing utilities for spam detection.

The philosophy is to keep preprocessing minimal so the model sees the same
signals that real messages contain. URLs, numbers, currency symbols, and words
like "free" or "call" are kept because they are legitimate spam indicators.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DEFAULT_MAX_FEATURES = 10000
DEFAULT_MIN_DF = 2
DEFAULT_MAX_DF = 0.95
DEFAULT_NGRAM_RANGE = (1, 2)


def build_tfidf_vectorizer(
    max_features: int = DEFAULT_MAX_FEATURES,
    min_df: int = DEFAULT_MIN_DF,
    max_df: float = DEFAULT_MAX_DF,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with sensible defaults for spam detection.

    The vectorizer is deliberately not fit here. Fit on the *training* messages
    only inside the training script or cached pipeline.
    """
    return TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
        sublinear_tf=True,
    )


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a reproducible train/test split with optional stratification.

    Stratification is enabled by default because the dataset is slightly
    imbalanced and we want both classes represented in each split.
    """
    if stratify:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["label"],
        )
    return train_test_split(df, test_size=test_size, random_state=random_state)


def get_default_example_messages() -> dict[str, str]:
    """Return a small set of labelled example messages for the Spam Detector page.

    These are provided so users can quickly test the model without typing.
    The labels are NOT used by the model — they're only for user context.
    """
    return {
        "spam_prize": "Congratulations! You have won a free gift. Click now to claim your prize.",
        "ham_meeting": "Hey, are we still on for the team meeting at 3 PM today?",
        "spam_verify": "URGENT: Your account has been compromised. Verify now at http://fake-link.com",
        "ham_appointment": "Reminder: Your dentist appointment is scheduled for tomorrow at 10 AM.",
        "spam_loan": "Get approved for a $5000 loan instantly! No credit check. Apply now!",
        "ham_friend": "Just wanted to check in and see how you're doing. Let me know when free!",
    }


def preprocess_message(message: str) -> str:
    """Lightweight message cleaning that preserves spam signals.

    This function is intentionally simple:
    - Strips leading/trailing whitespace
    - Normalizes multiple spaces to single space
    - Does NOT remove URLs, numbers, currency symbols, punctuation, or special tokens
    """
    if not isinstance(message, str):
        return ""
    msg = message.strip()
    msg = re.sub(r"\s+", " ", msg)
    return msg


def validate_message(message: str) -> tuple[bool, Optional[str]]:
    """Check if a user-entered message is valid for prediction.

    Returns (is_valid, error_message).
    """
    cleaned = preprocess_message(message)
    if not cleaned:
        return False, "Please enter a message to classify."
    if len(cleaned) > 10000:
        return False, "Message is too long (maximum 10,000 characters)."
    return True, None


def get_feature_names(vectorizer: TfidfVectorizer, top_n: int = 50) -> list[str]:
    """Return top-n feature names by overall TF-IDF score across training data.

    Useful for the Dataset Explorer to show the most characteristic tokens.
    """
    if not hasattr(vectorizer, "vocabulary_"):
        return []
    return list(vectorizer.get_feature_names_out())[:top_n]