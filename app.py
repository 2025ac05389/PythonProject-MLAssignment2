import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ==============================================================================
# Page configuration & Custom Styling
# ==============================================================================

st.set_page_config(
    page_title="Loan Approval Prediction ML Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; color: #1e3c72; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem !important; color: #495057; font-weight: 600; }
    .custom-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 22px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .custom-header h1 { color: #ffffff !important; margin: 0; font-size: 2.2rem; }
    .custom-header p { color: #e0e0e0 !important; margin-top: 5px; font-size: 1.0rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-header">
    <h1>🏦 Automated Loan Approval & Risk Intelligence Platform</h1>
    <p>BITS Pilani — Machine Learning Assignment 2 Interactive Analytics Workbench</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# Project paths & Artifact Loading
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib"
}

@st.cache_resource
def load_artifacts():
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    preprocessor = joblib.load(preprocessor_path)
    models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            model = joblib.load(path)
            if not hasattr(model, 'multi_class'):
                setattr(model, 'multi_class', 'auto')
            models[name] = model
    return preprocessor, models

try:
    preprocessor, models = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.info("Run `python model/train_models.py` first to generate required artifacts.")
    st.stop()

# Ensure required models exist
missing_models = [name for name in MODEL_FILES if name not in models]
if missing_models:
    st.error("Missing model artifacts: " + ", ".join(missing_models))
    st.info("Run `python model/train_models.py` and refresh application.")
    st.stop()

# ==============================================================================
# Sidebar Controls
# ==============================================================================

st.sidebar.header("⚙️ Model Selection")
selected_model_name = st.sidebar.selectbox("Choose Classifier", list(MODEL_FILES.keys()))
selected_model = models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Risk Threshold")
approval_threshold = st.sidebar.slider("Approval Cutoff Threshold (%)", min_value=30, max_value=90, value=50, step=5) / 100.0

st.sidebar.markdown("---")
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Evaluation CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Loaded uploaded dataset.")
else:
    default_test_path = os.path.join(BASE_DIR, "test_data.csv")
    default_train_path = os.path.join(BASE_DIR, "train.csv")
    if os.path.exists(default_test_path):
        df = pd.read_csv(default_test_path)
        st.sidebar.info("Loaded default `test_data.csv`.")
    elif os.path.exists(default_train_path):
        df = pd.read_csv(default_train_path)
        st.sidebar.info("Loaded default `train.csv`.")
    else:
        st.error("No dataset found.")
        st.stop()

# ==============================================================================
# Dataset Validation & Constraints
# ==============================================================================

st.subheader("📋 Dataset Overview & Constraint Verification")
st.dataframe(df, use_container_width=True, height=220)

feature_count = len([c for c in df.columns if c not in ['Loan_ID', 'Loan_Status']])
st.caption(f"Dataset contains **{len(df)}** rows and **{feature_count}** input features.")

if len(df) < 500:
    st.warning("⚠️ Dataset size is under 500 instances (Assignment guidelines recommend >= 500).")

if feature_count < 12:
    st.warning(f"⚠️ Dataset features ({feature_count}) are below recommended threshold (12 features).")

# ==============================================================================
# Feature Splitting & Preprocessing
# ==============================================================================

df_features = df.drop(columns=['Loan_ID'], errors='ignore')
has_target = 'Loan_Status' in df_features.columns

if has_target:
    X_test = df_features.drop(columns=['Loan_Status'])
    y_test = df_features['Loan_Status'].map({'Y': 1, 'N': 0, 1: 1, 0: 0})
else:
    X_test = df_features.copy()
    y_test = None

try:
    X_test_prep = preprocessor.transform(X_test)
except Exception as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()

# Probability Prediction
if hasattr(selected_model, "predict_proba"):
    y_proba = selected_model.predict_proba(X_test_prep)[:, 1]
else:
    y_proba = selected_model.decision_function(X_test_prep)

y_pred = (y_proba >= approval_threshold).astype(int)

# ==============================================================================
# Visual Evaluation Metrics (Heatmap & ROC Curve)
# ==============================================================================

if has_target and y_test.notnull().all():
    st.markdown("---")
    st.subheader(f"📊 Test Performance Metrics: {selected_model_name}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    c2.metric("AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
    c3.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
    c4.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
    c5.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
    c6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")

    st.markdown("---")
    col_vis1, col_vis2 = st.columns(2)

    with col_vis1:
        st.subheader("📍 Confusion Matrix (Seaborn Heatmap)")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                    xticklabels=['Predicted: Rejected (N)', 'Predicted: Approved (Y)'],
                    yticklabels=['Actual: Rejected (N)', 'Actual: Approved (Y)'])
        ax_cm.set_ylabel("Actual Ground Truth")
        ax_cm.set_xlabel("Model Prediction")
        st.pyplot(fig_cm)

    with col_vis2:
        st.subheader("📉 Receiver Operating Characteristic (ROC) Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 3.5))
        ax_roc.plot(fpr, tpr, color='#1e3c72', lw=2.5, label=f'AUC = {roc_auc_score(y_test, y_proba):.4f}')
        ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    # Classification Report Table
    st.subheader("📄 Detailed Classification Report")
    report = classification_report(y_test, y_pred, target_names=["Rejected (N)", "Approved (Y)"], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)

# ==============================================================================
# Model Comparison Matrix (All 5 Classifiers)
# ==============================================================================

if has_target and y_test.notnull().all():
    st.markdown("---")
    st.subheader("📈 Benchmark Comparison of All 5 Machine Learning Models")

    comparison_rows = []
    for model_name, model in models.items():
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X_test_prep)[:, 1]
        else:
            probability = model.decision_function(X_test_prep)

        prediction = (probability >= approval_threshold).astype(int)

        comparison_rows.append({
            "ML Model Name": model_name,
            "Accuracy": accuracy_score(y_test, prediction),
            "AUC": roc_auc_score(y_test, probability),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1": f1_score(y_test, prediction, zero_division=0),
            "MCC": matthews_corrcoef(y_test, prediction)
        })

    comparison_df = pd.DataFrame(comparison_rows)
    metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    comparison_df["Overall Score"] = comparison_df[metric_columns].mean(axis=1)

    st.dataframe(comparison_df.round(4), use_container_width=True)

    winner = comparison_df.loc[comparison_df["Overall Score"].idxmax()]
    st.success(f"🏆 Top Performing Classifier: **{winner['ML Model Name']}** (Mean Metric Score = {winner['Overall Score']:.4f})")

# ==============================================================================
# Prediction Export & Results Summary
# ==============================================================================

st.markdown("---")
st.subheader("🔮 Batch Loan Application Scoring Results")

results_df = df.copy()
results_df["Predicted_Loan_Status"] = np.where(y_proba >= approval_threshold, "Y (Approved)", "N (Rejected)")
results_df["Approval_Probability"] = (y_proba * 100).round(2).astype(str) + "%"

output_cols = ["Predicted_Loan_Status", "Approval_Probability"] + [c for c in df.columns if c not in ["Predicted_Loan_Status", "Approval_Probability"]]
st.dataframe(results_df[output_cols], use_container_width=True, height=300)

csv_output = results_df.to_csv(index=False)
st.download_button(
    label="⬇️ Download Prediction Results (CSV)",
    data=csv_output,
    file_name="loan_prediction_results.csv",
    mime="text/csv"
)
