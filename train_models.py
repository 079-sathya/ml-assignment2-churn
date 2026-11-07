"""
Train 5 classifiers on the Telco Customer Churn dataset and save them.

Run:  python train_models.py
Outputs:
  model/<name>.joblib      - full sklearn Pipeline (preprocessing + classifier)
  model/metrics.csv        - evaluation metrics on the held-out test split
  test_data.csv            - held-out test split (features + target) for the app
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "Churn"
DATA_PATH = Path("data/telco_churn.csv")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])                      # identifier, not a feature
    # TotalCharges has blank strings for customers with tenure 0 -> treat as 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    return df


df = load_and_clean(DATA_PATH)
X, y = df.drop(columns=[TARGET]), df[TARGET]

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = [c for c in X.columns if c not in numeric_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
])

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                            random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=25),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE),
}

rows = []
for name, clf in classifiers.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    # AUC needs a continuous score: probabilities where available, else SVM margin
    y_prob = (pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba")
              and hasattr(pipe.named_steps["clf"], "predict_proba")
              else pipe.decision_function(X_test))
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, MODEL_DIR / fname)
    print(f"saved {fname}")

metrics = pd.DataFrame(rows).set_index("Model").round(4)
metrics.to_csv(MODEL_DIR / "metrics.csv")
print("\n", metrics.to_markdown())

# Save the held-out test split for upload in the Streamlit app (target in original Yes/No form)
test_df = X_test.copy()
test_df[TARGET] = y_test.map({1: "Yes", 0: "No"})
test_df.to_csv("test_data.csv", index=False)

(MODEL_DIR / "feature_info.json").write_text(json.dumps(
    {"target": TARGET, "numeric": numeric_cols, "categorical": categorical_cols}, indent=2))
print(f"\ntrain={len(X_train)}  test={len(X_test)}")
