"""
Telco Customer Churn - Classifier Evaluation App
Loads 5 pre-trained sklearn pipelines from model/ and evaluates them on an
uploaded test CSV. No model is retrained here.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}
FEATURES = json.loads((MODEL_DIR / "feature_info.json").read_text())
TARGET = FEATURES["target"]
REQUIRED_COLS = FEATURES["numeric"] + FEATURES["categorical"]

st.set_page_config(page_title="Churn Classifier Lab", page_icon="📉", layout="wide")


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}


def prepare(df: pd.DataFrame):
    """Apply the same light cleaning used at training time; return X, y."""
    df = df.copy()
    df = df.drop(columns=[c for c in ["customerID"] if c in df.columns])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    y = df[TARGET]
    if y.dtype == object or str(y.dtype).startswith("str"):
        y = y.map({"Yes": 1, "No": 0})
    return df[REQUIRED_COLS], y.astype(int)


def scores(model, X):
    clf = model.named_steps["clf"]
    if getattr(clf, "probability", True) and hasattr(clf, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_score = scores(model, X)
    m = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_score) if y.nunique() > 1 else np.nan,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return m, y_pred, y_score


# ---------------- Sidebar ----------------
st.sidebar.title("📉 Churn Classifier Lab")
st.sidebar.caption("Telco Customer Churn · 5 pre-trained models")
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv",
                                    help="Must contain all feature columns and the 'Churn' column.")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES))
sample = BASE / "test_data.csv"
if sample.exists():
    st.sidebar.download_button("⬇ Download sample test_data.csv", sample.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")

st.title("Customer Churn Prediction — Model Evaluation")
st.write("Upload held-out test data to see how each pre-trained classifier performs. "
         "Models are loaded from disk; nothing is retrained on your upload.")

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.success(f"Using uploaded file: {uploaded.name}")
else:
    raw = pd.read_csv(sample)
    st.info("Showing results on the bundled test_data.csv. "
            "Upload your own test CSV from the sidebar to evaluate it instead.")

raw = pd.read_csv(uploaded)
missing = [c for c in REQUIRED_COLS + [TARGET] if c not in raw.columns]
if missing:
    st.error(f"Uploaded file is missing columns: {missing}")
    st.stop()

X, y = prepare(raw)
models = load_models()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(X):,}")
c2.metric("Features", len(REQUIRED_COLS))
c3.metric("Churn rate", f"{y.mean():.1%}")
with st.expander("Preview uploaded data"):
    st.dataframe(raw.head(20), width="stretch")

tab1, tab2 = st.tabs([f"🔍 {model_name}", "📊 Compare all models"])

with tab1:
    m, y_pred, y_score = evaluate(models[model_name], X, y)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("ROC curve")
        fpr, tpr, _ = roc_curve(y, y_score)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.plot(fpr, tpr, label=f"AUC = {m['AUC']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)

    st.subheader("Classification report")
    rep = classification_report(y, y_pred, target_names=["No churn", "Churn"], output_dict=True)
    st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

with tab2:
    table = pd.DataFrame({n: evaluate(mdl, X, y)[0] for n, mdl in models.items()}).T
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#c6f6d5"),
                 width="stretch")
    metric = st.radio("Metric to plot", list(table.columns), horizontal=True, index=5)
    fig, ax = plt.subplots(figsize=(8, 3))
    table[metric].sort_values().plot.barh(ax=ax, color="#3b82f6")
    ax.set_xlabel(metric)
    st.pyplot(fig)
