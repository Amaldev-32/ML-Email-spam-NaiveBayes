#!/usr/bin/env python3
"""Main application entry point for SpamShield AI - the Email Spam Detection app.

This is a Streamlit application that provides:

1. A spam detector with live predictions using a trained Naive Bayes model
2. An interactive dashboard showing model and dataset statistics
3. Dataset exploration tools for understanding the training data
4. Model performance analysis and visualizations
5. Educational content about machine learning concepts
6. Project information and documentation

The application uses a trained Multinomial Naive Bayes pipeline with TF-IDF
vectorization for email/spam classification, following a complete ML workflow.

It demonstrates how to deploy a machine learning model as an interactive web
application using Streamlit.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
import pandas as pd

# Import our ML modules
from src.data_loader import load_dataset, normalize_labels, safe_drop_bad_rows, inspect_dataset
from src.preprocessing import split_dataset, get_default_example_messages, validate_message
from src.model import load_model, predict_message, get_model_info, train_model, save_model
from src.evaluation import evaluate_model, format_classification_report, explain_metrics
from src.visualization import (
    make_class_distribution_chart,
    make_class_percentage_chart,
    make_message_length_distribution,
    make_word_count_distribution,
    make_top_words_chart,
    make_confusion_matrix_chart,
    make_classification_metrics_chart,
    make_roc_curve_chart,
    make_prediction_probability_chart,
    make_pipeline_diagram,
    make_metric_cards,
)

# Page configuration
st.set_page_config(
    page_title="SpamShield AI - Email Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
        color: #1e293b;
    }
    .metric-card {
        background-color: #f0fdfa;
        border: 1px solid #0d9488;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .spam-indicator {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 20px;
        color: #991b1b;
        font-weight: bold;
    }
    .ham-indicator {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 10px;
        padding: 20px;
        color: #166534;
        font-weight: bold;
    }
    .stButton > button {
        background-color: #0d9488;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #0f766e;
    }
    .warning-text {
        color: #dc2626;
        font-weight: 500;
    }
    .success-text {
        color: #16a34a;
        font-weight: 500;
    }
    .stTextArea > div > div > textarea {
        font-size: 16px;
        line-height: 1.5;
    }
    .info-box {
        background-color: #eff6ff;
        border: 1px solid #93c5fd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def display_metric_card(title: str, value: str, subtitle: str = ""):
    """Display a styled metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="color: #0d9488; margin-bottom: 10px;">{title}</h4>
            <h2 style="color: #1e293b; margin-bottom: 5px;">{value}</h2>
            <p style="color: #64748b; font-size: 14px;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_dashboard(state: Dict[str, Any]):
    """Show the main dashboard overview page."""
    st.header("📊 Dashboard Overview")

    dataset_stats = state.get("dataset_stats", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        display_metric_card(
            "Total Messages",
            f"{dataset_stats.get('total_rows', 0):,}",
            "from cleaned dataset"
        )

    with col2:
        display_metric_card(
            "Spam Messages",
            f"{dataset_stats.get('spam_count', 0):,}",
            f"{dataset_stats.get('spam_percentage', 0)*100:.1f}% of total"
        )

    with col3:
        display_metric_card(
            "Legitimate Messages",
            f"{dataset_stats.get('ham_count', 0):,}",
            f"{dataset_stats.get('ham_percentage', 0)*100:.1f}% of total"
        )

    with col4:
        if state.get("metrics"):
            acc = state["metrics"].get("accuracy", 0)
            display_metric_card(
                "Test Accuracy",
                f"{acc*100:.1f}%",
                "on held-out test set"
            )
        else:
            display_metric_card("Test Accuracy", "Not available", "Train model first")

    st.markdown("---")
    st.subheader("Visualizations")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if dataset_stats:
            st.plotly_chart(
                make_class_distribution_chart(
                    {"ham": dataset_stats.get("ham_count", 0),
                     "spam": dataset_stats.get("spam_count", 0)}
                ),
                use_container_width=True,
            )

    with chart_col2:
        if dataset_stats:
            st.plotly_chart(
                make_class_percentage_chart(
                    {"ham_pct": dataset_stats.get("ham_percentage", 0),
                     "spam_pct": dataset_stats.get("spam_percentage", 0)}
                ),
                use_container_width=True,
            )

    # ML Pipeline Visualization
    st.subheader("Machine Learning Pipeline")
    st.plotly_chart(make_pipeline_diagram(), use_container_width=True)

    # Model Performance (if available)
    if state.get("metrics"):
        st.subheader("Current Model Performance")
        st.plotly_chart(make_metric_cards(state["metrics"]), use_container_width=True)
    else:
        st.info("Model not trained yet. Go to Model Performance page to train and evaluate.")


def show_detector(state: Dict[str, Any]):
    """Show the interactive spam detector page."""
    st.header("🛡️ Spam Detector")

    if not state.get("model"):
        st.warning("Please load the model first from the Model Performance page.")
        return

    # Example messages
    with st.expander("📝 Example Messages (Click to use):", expanded=False):
        examples = get_default_example_messages()
        col1, col2 = st.columns(2)

        with col1:
            for key, msg in examples.items():
                if st.button(key.replace("_", " ").title(), key=f"ex_{key}", use_container_width=True):
                    st.session_state.user_input = msg

        with col2:
            st.markdown("**Instructions:** Click any example button above to use it as input.")
            st.markdown("**Tip:** The detector will show the model's confidence level.")

    # Main input area
    st.markdown("### Enter a message to classify:")
    user_input = st.text_area(
        label="Message",
        value=st.session_state.get("user_input", ""),
        placeholder="Enter or paste your message here...",
        height=150,
        max_chars=10000,
    )

    # Clear button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.user_input = ""
            st.rerun()

    # Check button
    if st.button("🔍 Check Message", use_container_width=True, type="primary"):
        is_valid, error_msg = validate_message(user_input)
        if not is_valid:
            st.error(f"⚠️ {error_msg}")
        else:
            with st.spinner("Classifying message..."):
                result = predict_message(state["model"], user_input)

                if result["prediction"] == "spam":
                    st.markdown(
                        f"""
                        <div class="spam-indicator">
                            <h3 style="color: #dc2626; margin-bottom: 15px;">🚨 SPAM DETECTED</h3>
                            <p><strong>Prediction:</strong> This message is classified as <strong>SPAM</strong></p>
                            <p><strong>Spam Probability:</strong> {result["spam_probability"]:.2%}</p>
                            <p><strong>Legitimate Probability:</strong> {result["ham_probability"]:.2%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="ham-indicator">
                            <h3 style="color: #16a34a; margin-bottom: 15px;">✅ LEGITIMATE MESSAGE</h3>
                            <p><strong>Prediction:</strong> This message is classified as <strong>LEGITIMATE</strong></p>
                            <p><strong>Spam Probability:</strong> {result["spam_probability"]:.2%}</p>
                            <p><strong>Legitimate Probability:</strong> {result["ham_probability"]:.2%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.plotly_chart(
                    make_prediction_probability_chart(
                        result["ham_probability"], result["spam_probability"]
                    ),
                    use_container_width=True,
                )

                st.markdown("### 📊 Model Explanation")
                confidence = max(result["spam_probability"], result["ham_probability"])
                if confidence > 0.8:
                    st.markdown("This message is classified with high confidence.")
                elif confidence > 0.6:
                    st.markdown("This message has moderate confidence.")
                else:
                    st.markdown("This message has low confidence. Consider manual review.")

                st.markdown("""
                ### ⚠️ Important Notice
                This classification is based on a statistical model trained on historical data.
                It's not a guarantee of message safety or legitimacy. Use your judgment and
                be cautious with unexpected messages, even if classified as legitimate.
                """, unsafe_allow_html=True)

    if user_input:
        st.markdown(f"**Characters:** {len(user_input)} | **Words:** {len(user_input.split())}")


def show_explorer(state: Dict[str, Any]):
    """Show the dataset explorer page."""
    st.header("📋 Dataset Explorer")

    if not state.get("dataset_df"):
        st.warning("Please load the dataset first.")
        return

    df = state["dataset_df"]
    meta = state.get("data_meta", {})

    st.subheader("Dataset Overview")
    overview_col1, overview_col2, overview_col3 = st.columns(3)

    with overview_col1:
        display_metric_card(
            "Total Messages",
            f"{len(df):,}",
            f"after removing {meta.get('rows_dropped', 0)} invalid rows"
        )

    with overview_col2:
        display_metric_card(
            "Spam Messages",
            f"{meta.get('missing_messages', 0):,}",
            "original spam count from dataset"
        )

    with overview_col3:
        display_metric_card(
            "Duplicates Removed",
            f"{meta.get('duplicates_removed', 0):,}",
            "to ensure clean data"
        )

    st.markdown("---")
    st.subheader("Class Distribution")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.plotly_chart(
            make_class_distribution_chart(
                {"ham": df["label"].value_counts().get("ham", 0),
                 "spam": df["label"].value_counts().get("spam", 0)}
            ),
            use_container_width=True,
        )

    with chart_col2:
        ham_pct = (df["label"] == "ham").mean() * 100
        spam_pct = (df["label"] == "spam").mean() * 100
        st.plotly_chart(
            make_class_percentage_chart(
                {"ham_pct": ham_pct / 100, "spam_pct": spam_pct / 100}
            ),
            use_container_width=True,
        )

    st.subheader("Message Length Distribution")
    st.plotly_chart(make_message_length_distribution(df, "message", "label"), use_container_width=True)

    st.subheader("Word Count Distribution")
    st.plotly_chart(make_word_count_distribution(df, "message", "label"), use_container_width=True)

    st.markdown("---")
    st.subheader("Most Frequent Words")
    word_col1, word_col2 = st.columns(2)

    with word_col1:
        st.markdown("**Most Frequent Words in Spam Messages:**")
        st.plotly_chart(make_top_words_chart(df, "spam", "message", "label", top_n=15), use_container_width=True)

    with word_col2:
        st.markdown("**Most Frequent Words in Legitimate Messages:**")
        st.plotly_chart(make_top_words_chart(df, "ham", "message", "label", top_n=15), use_container_width=True)

    with st.expander("📊 View Sample Data", expanded=False):
        st.markdown("**Sample of dataset messages:**")
        sample_size = min(20, len(df))
        sample_df = df.head(sample_size)
        st.dataframe(
            sample_df,
            column_config={
                "label": "Class",
                "message": st.column_config.TextColumn("Message", width="large"),
            },
            hide_index=True,
        )


def show_performance(state: Dict[str, Any]):
    """Show the model performance evaluation page."""
    st.header("📈 Model Performance")

    # Model info
    if state.get("model_info"):
        st.subheader("Model Information")
        info = state["model_info"]
        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.markdown(f"**Model Type:** {info.get('model_type', 'N/A')}")
            st.markdown(f"**Alpha (Smoothing):** {info.get('alpha', 'N/A')}")
            st.markdown(f"**Vocabulary Size:** {info.get('n_features', 'N/A')}")

        with info_col2:
            st.markdown(f"**Classes:** {', '.join(info.get('classes', []))}")
            train_df = state.get("train_df")
            test_df = state.get("test_df")
            st.markdown(f"**Training Samples:** {len(train_df) if train_df is not None else 'N/A'}")
            st.markdown(f"**Test Samples:** {len(test_df) if test_df is not None else 'N/A'}")

        st.markdown("---")

    # Evaluation metrics
    if not state.get("metrics"):
        st.info("No evaluation metrics available. Please train and evaluate the model.")

        if st.button("🚀 Train and Evaluate Model", type="primary"):
            with st.spinner("Training model and calculating metrics..."):
                try:
                    train_df = state.get("train_df")
                    test_df = state.get("test_df")

                    if train_df is None or test_df is None:
                        st.error("Training data not available. Reload the page.")
                        return

                    pipeline = train_model(train_df)
                    metrics = evaluate_model(pipeline, test_df)
                    metrics["train_samples"] = len(train_df)

                    save_model(pipeline)

                    state.update({
                        "model": pipeline,
                        "metrics": metrics,
                        "model_info": get_model_info(pipeline),
                    })

                    st.success("Model trained and evaluated successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error training model: {e}")
    else:
        metrics = state["metrics"]

        st.subheader("Key Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            accuracy = metrics.get("accuracy", 0)
            display_metric_card("Accuracy", f"{accuracy*100:.2f}%", "Test set accuracy")

        with metric_col2:
            precision = metrics.get("precision_spam", 0)
            display_metric_card("Spam Precision", f"{precision*100:.2f}%", "Of predicted spam, actual spam")

        with metric_col3:
            recall = metrics.get("recall_spam", 0)
            display_metric_card("Spam Recall", f"{recall*100:.2f}%", "Of actual spam, correctly detected")

        with metric_col4:
            f1 = metrics.get("f1_spam", 0)
            display_metric_card("F1-Score", f"{f1*100:.2f}%", "Harmonic mean of precision/recall")

        st.markdown("---")

        st.subheader("Confusion Matrix")
        cm = metrics.get("confusion_matrix", {})
        st.plotly_chart(make_confusion_matrix_chart(cm.get("matrix", [[0, 0], [0, 0]])), use_container_width=True)

        st.subheader("Classification Report")
        report_df = format_classification_report(metrics.get("classification_report", {}))
        st.dataframe(report_df, use_container_width=True)

        if metrics.get("roc_auc"):
            st.subheader("ROC Curve")
            roc_data = {
                "fpr": metrics.get("roc_curve", {}).get("fpr"),
                "tpr": metrics.get("roc_curve", {}).get("tpr"),
                "auc": metrics.get("roc_auc"),
            }
            st.plotly_chart(make_roc_curve_chart(roc_data), use_container_width=True)

        st.markdown("---")
        st.subheader("Metrics Explanation")
        explanations = explain_metrics()
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown(f"**True Positives:** {explanations.get('true_positives', 'N/A')}")
            st.markdown(f"**True Negatives:** {explanations.get('true_negatives', 'N/A')}")
            st.markdown(f"**False Positives:** {explanations.get('false_positives', 'N/A')}")
            st.markdown(f"**False Negatives:** {explanations.get('false_negatives', 'N/A')}")

        with exp_col2:
            st.markdown(f"**Accuracy:** {explanations.get('accuracy', 'N/A')}")
            st.markdown(f"**Precision:** {explanations.get('precision', 'N/A')}")
            st.markdown(f"**Recall:** {explanations.get('recall', 'N/A')}")
            st.markdown(f"**F1-Score:** {explanations.get('f1_score', 'N/A')}")


def show_how_it_works():
    """Show the educational page explaining how the system works."""
    st.header("📚 How It Works")

    st.markdown("""
    ## Supervised Learning

    The SpamShield AI system uses **supervised learning** to classify messages.
    It learns from a large dataset of already labeled messages (ham and spam) and
    applies this knowledge to new, unseen messages.

    ## Text Vectorization

    Machine learning algorithms work with numbers, not text. We convert messages into
    numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

    **Term Frequency (TF):** How often a word appears in a message
    **Inverse Document Frequency (IDF):** How distinctive that word is across all messages

    TF-IDF gives more weight to words that are:
    - Common in a specific message (high TF)
    - Rare across all messages (high IDF)

    For example, words like "free", "winner", "prize" might be strong spam indicators
    if they appear frequently in spam messages but rarely in ham.

    ## Naive Bayes Algorithm

    We use **Multinomial Naive Bayes**, which applies **Bayes' theorem** with the
    assumption that features (word occurrences) are conditionally independent given
    the class.

    **The Formula:**

    P(Class|Features) = P(Features|Class) × P(Class) / P(Features)

    Where:
    - P(Class|Features): Probability of class given the message
    - P(Features|Class): Probability of observing those features in that class
    - P(Class): Prior probability of the class

    ## Training and Testing

    The dataset is split into:
    - **Training set:** Used to train the model
    - **Test set:** Used to evaluate performance on unseen data

    This separation prevents overfitting and gives an honest assessment.

    ## Evaluation Metrics

    **Accuracy:** Overall fraction of correct predictions

    **Precision (Spam):** Of messages predicted as spam, how many are actually spam
    **Recall (Spam):** Of actual spam messages, how many were caught

    **F1-Score:** Balance between precision and recall

    ## Limitations

    - The model doesn't understand message meaning
    - Spam patterns evolve over time
    - Ambiguous messages might be misclassified
    - Class imbalance could affect performance
    - Performance depends on training data quality
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("ML Pipeline Visualization")
    st.plotly_chart(make_pipeline_diagram(), use_container_width=True)


def show_about():
    """Show the about/project information page."""
    st.header("ℹ️ About Project")

    st.markdown("""
    ## Email Spam Detection Using Naive Bayes

    ### Application Name
    **SpamShield AI**

    ### Project Type
    Machine Learning — Supervised Text Classification

    ### Algorithm
    Multinomial Naive Bayes

    ### Feature Extraction
    TF-IDF (Term Frequency-Inverse Document Frequency)

    ### Dataset
    UCI SMS Spam Collection

    ### Programming Language
    Python 3.x

    ### Libraries Used
    - **scikit-learn:** Machine learning algorithms and evaluation metrics
    - **pandas:** Data manipulation and analysis
    - **numpy:** Numerical operations
    - **Streamlit:** Web application framework
    - **Plotly:** Interactive visualizations

    ### Project Objectives

    1. **Demonstrate ML Workflow:** Show a complete machine learning pipeline from data loading to deployment

    2. **Educational Value:** Provide a beginner-friendly example suitable for college ML courses and viva presentations

    3. **Practical Utility:** Deliver a working spam detection tool that users can interact with

    4. **Technical Excellence:** Implement best practices for code organization, testing, and deployment

    ### Dataset Attribution

    This project uses the **UCI SMS Spam Collection dataset**, available at:
    https://archive.ics.uci.edu/static/public/246/smsspamcollection.zip

    The dataset contains labeled SMS messages (ham and spam) collected for research purposes.

    ### Intended Use

    - **Educational:** For teaching machine learning concepts
    - **Demonstration:** To showcase text classification capabilities
    - **Research:** As a baseline for spam detection research

    ### Important Notes

    - This is a course project, not a production system
    - Model accuracy may be limited and should not be used for critical decisions
    - Performance depends on the quality and recency of training data

    ### Future Improvements

    - Advanced preprocessing techniques
    - Ensemble methods for better accuracy
    - Integration with email providers
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    **Project Details:**
    - Type: Machine Learning Course Project
    - Target Audience: College students and ML practitioners
    - Deployment: Streamlit Community Cloud
    - License: MIT (proposed)

    **Version:** 1.0.0
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("Built with ❤️ for educational purposes", unsafe_allow_html=True)


def main_app():
    """Main application entry point with sidebar navigation."""
    # Initialize session state
    if "app_state" not in st.session_state:
        st.session_state.app_state = {
            "model": None,
            "dataset_df": None,
            "train_df": None,
            "test_df": None,
            "metrics": None,
            "model_info": None,
            "dataset_stats": None,
            "data_meta": None,
            "dataset_loaded": False,
            "user_input": "",
        }
        st.session_state.current_page = "dashboard"
    else:
        state = st.session_state.app_state
        state.setdefault("model", None)
        state.setdefault("dataset_df", None)
        state.setdefault("train_df", None)
        state.setdefault("test_df", None)
        state.setdefault("metrics", None)
        state.setdefault("model_info", None)
        state.setdefault("dataset_stats", None)
        state.setdefault("data_meta", None)
        state.setdefault("dataset_loaded", False)
        state.setdefault("user_input", "")
        st.session_state.setdefault("current_page", "dashboard")

    state = st.session_state.app_state

    # Sidebar navigation
    st.sidebar.title("🗺️ Navigation")
    st.sidebar.markdown("""
    ## SpamShield AI
    <p style="color: #64748b; font-size: 14px;">Intelligent Message Classification Powered by Naive Bayes</p>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")

    nav_pages = [
        ("📊 Dashboard", "dashboard"),
        ("🛡️ Spam Detector", "detector"),
        ("📋 Dataset Explorer", "explorer"),
        ("📈 Model Performance", "performance"),
        ("📚 How It Works", "how_it_works"),
        ("ℹ️ About Project", "about"),
    ]

    for label, page_id in nav_pages:
        if st.sidebar.button(label, key=f"nav_{page_id}", use_container_width=True):
            st.session_state.current_page = page_id

    # Main title
    st.title("🛡️ SpamShield AI")
    st.markdown("Intelligent Message Classification Powered by Naive Bayes")

    # Load data on first run
    if not state["dataset_loaded"]:
        with st.spinner("Loading dataset and model..."):
            try:
                raw_df = load_dataset()
                df = normalize_labels(raw_df)
                df, meta = safe_drop_bad_rows(df)
                train_df, test_df = split_dataset(df)

                # Load existing model or train new one
                model_path = Path("models/spam_model.joblib")
                if model_path.exists():
                    model = load_model(model_path)
                    # Load metrics if they exist
                    metrics_path = Path("models/model_metrics.json")
                    if metrics_path.exists():
                        import json
                        with open(metrics_path) as f:
                            metrics = json.load(f)
                            metrics["train_samples"] = len(train_df)
                    else:
                        metrics = None
                else:
                    model = train_model(train_df)
                    save_model(model)
                    metrics = None

                state.update({
                    "model": model,
                    "dataset_df": df,
                    "train_df": train_df,
                    "test_df": test_df,
                    "model_info": get_model_info(model),
                    "dataset_stats": inspect_dataset(df),
                    "data_meta": meta,
                    "dataset_loaded": True,
                    "metrics": metrics,
                })
                st.success("Dataset loaded successfully!")

            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.stop()

    # Page routing
    page = st.session_state.current_page

    if page == "dashboard":
        show_dashboard(state)
    elif page == "detector":
        show_detector(state)
    elif page == "explorer":
        show_explorer(state)
    elif page == "performance":
        show_performance(state)
    elif page == "how_it_works":
        show_how_it_works()
    elif page == "about":
        show_about()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #94a3b8; font-size: 14px;">
            <p>© 2026 SpamShield AI Project | Machine Learning Course Project</p>
            <p>Dataset: UCI SMS Spam Collection | Model: Multinomial Naive Bayes</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main_app()