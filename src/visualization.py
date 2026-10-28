"""Visualization helpers for SpamShield AI charts and dashboards."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .evaluation import explain_metrics


def _apply_common_style(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply shared layout styling consistent with the app theme."""
    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#0f172a"},
        },
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#33415b"),
        title_font=dict(color="#0d9488"),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    return fig


def make_class_distribution_chart(distribution: Dict[str, int]) -> go.Figure:
    """Bar chart comparing spam vs ham counts."""
    classes = ["ham", "spam"]
    counts = [distribution.get(c, 0) for c in classes]
    colors = ["#22c55e", "#ef4444"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Ham (Legitimate)", "Spam"],
                y=counts,
                marker_color=colors,
                text=counts,
                textposition="outside",
            )
        ]
    )
    _apply_common_style(fig, "Message Class Distribution")
    fig.update_layout(
        xaxis_title="Message Class",
        yaxis_title="Number of Messages",
        showlegend=False,
    )
    return fig


def make_class_percentage_chart(distribution: Dict[str, float]) -> go.Figure:
    """Pie chart showing ham/spam percentages."""
    labels = ["Ham (Legitimate)", "Spam"]
    values = [distribution.get("ham_pct", 0), distribution.get("spam_pct", 0)]
    colors = ["#22c55e", "#ef4444"]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                textinfo="label+percent+value",
                hole=0.4,
            )
        ]
    )
    _apply_common_style(fig, "Class Percentage")
    return fig


def make_message_length_distribution(
    df: pd.DataFrame, message_col: str = "message", label_col: str = "label"
) -> go.Figure:
    """Box/violin plot of message lengths by class."""
    df = df.copy()
    df["length"] = df[message_col].astype(str).str.len()

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            x=df["length"],
            y=df[label_col],
            box_visible=True,
            meanline_visible=True,
            line_color="#0d9488",
            name="Ham",
        )
    )
    fig = px.violin(
        df,
        x="length",
        color=label_col,
        box=True,
        points="all",
        color_discrete_map={"ham": "#22c55e", "spam": "#ef4444"},
        labels={"length": "Message Length (characters)", "label": "Class"},
        title="Message Length Distribution by Class",
    )
    _apply_common_style(fig, "Message Length Distribution by Class")
    fig.update_layout(
        xaxis_title="Message Length (characters)",
        yaxis_title="Count",
        legend_title_text="Class",
    )
    return fig


def make_word_count_distribution(
    df: pd.DataFrame, message_col: str = "message", label_col: str = "label"
) -> go.Figure:
    """Distribution of word counts by class."""
    df = df.copy()
    df["word_count"] = df[message_col].astype(str).apply(lambda x: len(x.split()))

    fig = px.histogram(
        df,
        x="word_count",
        color=label_col,
        marginal="box",
        barmode="overlay",
        nbins=50,
        color_discrete_map={"ham": "#22c55e", "spam": "#ef4444"},
        labels={"word_count": "Number of Words", "label": "Class"},
        title="Word Count Distribution by Class",
    )
    _apply_common_style(fig, "Word Count Distribution by Class")
    fig.update_layout(
        xaxis_title="Number of Words",
        yaxis_title="Frequency",
        legend_title_text="Class",
    )
    return fig


def make_top_words_chart(
    df: pd.DataFrame,
    label: str,
    message_col: str = "message",
    label_col: str = "label",
    top_n: int = 20,
    stop_words: Optional[frozenset] = None,
) -> go.Figure:
    """Horizontal bar chart of the top-N most frequent meaningful words for a class."""
    if stop_words is None:
        stop_words = frozenset(
            [
                "the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
                "or", "but", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "i", "you", "he",
                "she", "it", "we", "they", "them", "this", "that", "with",
                "from", "as", "by", "not", "no", "my", "your", "his", "her",
                "its", "our", "their", "what", "which", "who", "how",
                "can", "will", "just", "than", "so", "if", "out", "up",
                "about", "above", "below", "call", "free", "txt", "text",
            ]
        )

    class_df = df[df[label_col] == label]
    all_words = " ".join(class_df[message_col].astype(str).tolist()).lower()
    words = [w.strip(".,!?;:\"'()[]{}") for w in all_words.split()]
    words = [w for w in words if w and w not in stop_words and len(w) > 1]

    word_counts = pd.Series(words).value_counts().head(top_n).sort_values()

    fig = go.Figure(
        data=[
            go.Bar(
                x=word_counts.values,
                y=word_counts.index,
                orientation="h",
                marker_color="#0d9488",
                text=word_counts.values,
                textposition="outside",
            )
        ]
    )
    _apply_common_style(fig, f"Top {top_n} Words in {label.title()} Messages")
    fig.update_layout(
        xaxis_title="Frequency",
        yaxis_title="Word",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def make_confusion_matrix_chart(
    cm: List[List[int]],
    labels: Optional[List[str]] = None,
) -> go.Figure:
    """Render a labelled confusion matrix heatmap."""
    if labels is None:
        labels = ["Ham", "Spam"]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[[0, "#dcfce7"], [1, "#86efac"], [1, "#fca5a5"]],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 18},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    _apply_common_style(fig, "Confusion Matrix")
    fig.update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def make_classification_metrics_chart(metrics: Dict[str, float]) -> go.Figure:
    """Grouped bar chart comparing precision, recall, F1 for ham and spam."""
    categories = ["Precision", "Recall", "F1-Score"]
    spam_vals = [metrics.get("precision_spam", 0), metrics.get("recall_spam", 0), metrics.get("f1_spam", 0)]
    ham_vals = [
        metrics.get("classification_report", {}).get("ham", {}).get("precision", 0),
        metrics.get("classification_report", {}).get("ham", {}).get("recall", 0),
        metrics.get("classification_report", {}).get("ham", {}).get("f1-score", 0),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Ham",
            x=categories,
            y=ham_vals,
            marker_color="#22c55e",
            text=[round(v, 4) for v in ham_vals],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Spam",
            x=categories,
            y=spam_vals,
            marker_color="#ef4444",
            text=[round(v, 4) for v in spam_vals],
            textposition="outside",
        )
    )
    _apply_common_style(fig, "Classification Metrics Comparison")
    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Score",
        barmode="group",
        yaxis_range=[0, 1.05],
    )
    return fig


def make_roc_curve_chart(roc_data: Optional[Dict] = None) -> go.Figure:
    """Plot the ROC curve if data is available."""
    fig = go.Figure()
    if roc_data and roc_data.get("fpr") is not None:
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        auc = roc_data.get("auc", 0)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC Curve (AUC = {auc:.4f})" if auc else "ROC Curve",
                line=dict(color="#0d9488", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="#94a3b8", width=1, dash="dash"),
            )
        )
    else:
        fig.add_annotation(
            text="ROC curve not available",
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16, color="#94a3b8"),
        )
    _apply_common_style(fig, "ROC Curve")
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.05],
    )
    return fig


def make_prediction_probability_chart(
    ham_prob: float, spam_prob: float
) -> go.Figure:
    """Bar chart showing ham vs spam probabilities for a prediction."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Ham", "Spam"],
                y=[ham_prob, spam_prob],
                marker_color=["#22c55e", "#ef4444"],
                text=[f"{ham_prob:.4f}", f"{spam_prob:.4f}"],
                textposition="outside",
            )
        ]
    )
    _apply_common_style(fig, "Prediction Probabilities")
    fig.update_layout(
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        showlegend=False,
    )
    return fig


def make_pipeline_diagram() -> go.Figure:
    """Create a simple visual representation of the ML pipeline.

    Input Messages -> TF-IDF -> MultinomialNB -> Prediction
    """
    fig = go.Figure()

    steps = [
        {"name": "Input\nMessage", "x": 0},
        {"name": "TF-IDF\nVectorizer", "x": 1},
        {"name": "MultinomialNB\nNaive Bayes", "x": 2},
        {"name": "Prediction\n(Ham or Spam)", "x": 3},
    ]

    for i, step in enumerate(steps):
        fig.add_shape(
            type="rect",
            x0=step["x"] - 0.35,
            x1=step["x"] + 0.35,
            y0=-0.3,
            y1=0.3,
            fillcolor="#f0fdfa",
            line=dict(color="#0d9488", width=2),
            layer="below",
        )
        fig.add_annotation(
            x=step["x"],
            y=0,
            text=step["name"],
            showarrow=False,
            font=dict(size=13, color="#0f172a"),
        )

    # Arrows between steps
    for i in range(len(steps) - 1):
        fig.add_annotation(
            x=steps[i]["x"] + 0.35,
            y=0,
            ax=steps[i + 1]["x"] - 0.35,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="#0d9488",
        )

    fig.update_layout(
        title={
            "text": "Machine Learning Pipeline",
            "x": 0.5,
            "font": {"size": 18},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        xaxis=dict(
            range=[-0.5, 3.5],
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        yaxis=dict(
            range=[-0.6, 0.6],
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        margin=dict(l=40, r=40, t=60, b=20),
    )
    return fig


def make_metric_cards(metrics: Dict[str, float]) -> go.Figure:
    """Create a grouped bar for metric cards-style visualization."""
    metric_names = ["Accuracy", "Precision (Spam)", "Recall (Spam)", "F1-Score (Spam)"]
    values = [
        metrics.get("accuracy", 0),
        metrics.get("precision_spam", 0),
        metrics.get("recall_spam", 0),
        metrics.get("f1_spam", 0),
    ]
    colors = ["#3b82f6", "#f59e0b", "#8b5cf6", "#0d9488"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=metric_names,
                y=values,
                marker_color=colors,
                text=[round(v, 4) for v in values],
                textposition="outside",
            )
        ]
    )
    _apply_common_style(fig, "Model Performance Metrics")
    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis_range=[0, 1.1],
        showlegend=False,
    )
    return fig