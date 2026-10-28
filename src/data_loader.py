"""Dataset loading utilities for the SpamShield AI project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from urllib.request import urlretrieve

# Official dataset source: UCI Machine Learning Repository — SMS Spam Collection.
# The raw file is tab-separated ("\t") and has a header line of two columns: "label" and "message".
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/246/smsspamcollection.zip"
)
# The zip archive contains SMSSpamCollection inside it.
ZIP_FILENAME = "SMSSpamCollection.zip"
RAW_CSV_FILENAME = "SMSSpamCollection"

LOCAL_DATASETS_DIR = Path("data/datasets")

LABEL_MAP: Dict[str, str] = {
    "ham": "ham",
    "spam": "spam",
}

# Ordered class names used across the application for spam-positive defaults.
DEFAULT_CLASSES = ["ham", "spam"]


def default_dataset_path() -> Path:
    """Return the default path to the downloaded dataset CSV.

    The application downloads the official UCI SMS Spam Collection into a local
    datasets folder on first run. If the file already exists locally, downloading
    is skipped so repeated Streamlit reruns stay fast.
    """
    return LOCAL_DATASETS_DIR / RAW_CSV_FILENAME


def download_dataset() -> str:
    """Download the official SMS Spam Collection dataset if missing.

    Returns the local filename of the downloaded CSV. Raises FileNotFoundError if
    a previously cached file is present but corrupt; otherwise shows a warning.
    Never fabricates data on download failure.
    """
    target = LOCAL_DATASETS_DIR / ZIP_FILENAME
    csv_target = LOCAL_DATASETS_DIR / RAW_CSV_FILENAME
    if csv_target.exists():
        return str(csv_target)

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATASET_URL, target)

    import zipfile

    with zipfile.ZipFile(target, "r") as zf:
        zf.extractall(target.parent)
    return str(csv_target)


def load_raw_dataset(url: str = DATASET_URL) -> pd.DataFrame:
    """Download and load the raw UCI SMS Spam Collection file.

    This is the fallback when the user supplies a local copy by hand or when the
    automatic download path is blocked.
    """
    filename = download_dataset() if not (LOCAL_DATASETS_DIR / RAW_CSV_FILENAME).exists() else None
    path = LOCAL_DATASETS_DIR / RAW_CSV_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Could not locate the dataset at {path}. "
            f"Please supply it manually or allow network access to download it. "
            f"Expected columns 'label' and 'message', and values 'ham'/'spam'."
        )
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    if df.empty:
        raise ValueError("Dataset loaded but it was empty.")
    return df


def load_dataset() -> pd.DataFrame:
    """Load the dataset following this priority:

    1. A locally supplied dataset file under ``data/`` or ``data/datasets/``.
    2. The official UCI SMS Spam Collection auto-downloaded into ``data/datasets/``.
    3. Raise a clear error explaining manual supply.

    Rows with missing labels or messages are dropped before any processing
    pipeline begins; that decision is recorded as a meta flag rather than hidden.
    """
    candidate = [
        LOCAL_DATASETS_DIR / RAW_CSV_FILENAME,
        Path("data/SMSSpamCollection"),
        Path("data/datasets/SMSSpamCollection"),
    ]
    for path in candidate:
        if path.exists():
            df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
            break
    else:
        download_dataset()
        df = pd.read_csv(LOCAL_DATASETS_DIR / RAW_CSV_FILENAME, sep="\t", header=None, names=["label", "message"])

    # Ensure both classes are actually present.
    unique_labels = set(df["label"].astype(str).str.lower())
    if len(unique_labels) < 2:
        raise ValueError(
            "Dataset contains only one label class. Expected at least two distinct labels "
            "(ham and spam)."
        )

    return df.copy()


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy source labels to normalized ``ham``/``spam`` values.

    Also normalizes case and trims whitespace so that classes are consistent.
    """
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    valid = {"ham", "spam"}
    missing = [lbl for lbl in df["label"].unique() if lbl not in valid]
    if missing:
        raise ValueError(f"Encountered invalid labels after normalization: {missing}")
    return df.assign(label=df["label"].map(LABEL_MAP))


def safe_drop_bad_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Drop rows with missing or empty messages, reporting counts removed.

    The reported statistics explain how many data were lost because of cleaning,
    which the UI surfaces under the Dataset Explorer.
    """
    before = len(df)
    df = df.dropna(subset=["label", "message"]).copy()
    df["message"] = df["message"].fillna("")
    msg_empty = int((df["message"].astype(str).str.strip() == "").sum())
    removed = before - len(df)
    meta = {
        "original_rows": before,
        "rows_dropped": removed,
        "missing_messages": msg_empty,
        "duplicates_removed": int(df.duplicated(subset=["label", "message"]).sum()),
    }
    return df, meta


def inspect_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """Return structural statistics about the cleaned dataset for the dashboard.

    Does not perform any training or tuning decisions; only describes what was
    seen after ``safe_drop_bad_rows``.
    """
    total = len(df)
    n_spam = int((df["label"] == "spam").sum())
    n_ham = int((df["label"] == "ham").sum())
    n_duplicates = int(df.duplicated(subset=["label", "message"]).sum())
    return {
        "total_rows": total,
        "spam_count": n_spam,
        "ham_count": n_ham,
        "duplicate_count": n_duplicates,
        "spam_percentage": round(n_spam / total, 4) if total else 0.0,
        "ham_percentage": round(n_ham / total, 4) if total else 0.0,
    }
