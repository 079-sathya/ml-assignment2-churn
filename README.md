# Telco Customer Churn — Classification Model Comparison

**Live app:** <PASTE_STREAMLIT_APP_URL>
**Repository:** <https://github.com/079-sathya/ml-assignment2-churn>

## a. Problem Statement

A telecom provider loses revenue every time a subscriber cancels. The goal is to predict, from a
customer's account profile, subscribed services and billing details, whether that customer will
**churn** (leave) — a binary classification problem. Five classifiers are trained on the same data,
evaluated with six metrics, and deployed in a Streamlit app that scores uploaded test data using the
saved models (no retraining).

## b. Dataset Description

| Item | Detail |
|---|---|
| Name | Telco Customer Churn (IBM sample dataset, published on Kaggle: `blastchar/telco-customer-churn`) |
| Instances | 7,043 customers |
| Features | 19 input features (after dropping `customerID`) |
| Target | `Churn` — Yes (1,869 · 26.5%) / No (5,174 · 73.5%) |
| Split | 80/20 stratified train/test (5,634 / 1,409), `random_state=42` |

**Features**
- Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Account: `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod`
- Services: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`,
  `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- Charges: `MonthlyCharges`, `TotalCharges`

**Preprocessing**
- `customerID` dropped (identifier).
- `TotalCharges` is stored as text with 11 blank values (customers with tenure 0); converted to numeric and blanks set to 0.
- Numeric columns standardised (`StandardScaler`); 16 categorical columns one-hot encoded.
- Preprocessing is packaged inside each saved `Pipeline`, so the app applies exactly the training-time transformation.

## c. Repository Structure

```
├── app.py                 # Streamlit app
├── train_models.py        # trains & saves all 5 models, writes metrics + test split
├── requirements.txt
├── README.md
├── test_data.csv          # 20% held-out test split (features + Churn)
├── data/telco_churn.csv   # full raw dataset
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics.csv
    └── feature_info.json
```

Run locally:
```bash
pip install -r requirements.txt
python train_models.py      # optional: regenerates models/ and test_data.csv
streamlit run app.py
```

## d. Models Used and Evaluation

Models: Logistic Regression, Decision Tree (max_depth=6, min_samples_leaf=20), K-Nearest Neighbours
(k=25), Gaussian Naive Bayes, Support Vector Machine (RBF kernel, C=1). AUC for the SVM uses its
decision-function margin; all other models use predicted probabilities.

### Comparison Table (test set, n = 1,409)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8055 | 0.8420 | 0.6572 | 0.5588 | 0.6040 | 0.4790 |
| Decision Tree | 0.7970 | 0.8374 | 0.6679 | 0.4679 | 0.5503 | 0.4357 |
| KNN | 0.7842 | 0.8273 | 0.5989 | 0.5668 | 0.5824 | 0.4374 |
| Naive Bayes | 0.6955 | 0.8075 | 0.4597 | 0.8396 | 0.5941 | 0.4268 |
| Support Vector Machine | 0.7899 | 0.7928 | 0.6373 | 0.4840 | 0.5502 | 0.4232 |

Reference point: always predicting "No churn" gives 73.5% accuracy, so accuracy alone is
misleading here — MCC, F1 and AUC are the more informative metrics.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on 4 of 6 metrics (Accuracy 0.806, AUC 0.842, F1 0.604, MCC 0.479). After one-hot encoding, churn drivers such as month-to-month contract, fibre-optic internet, electronic-check payment and short tenure act roughly additively on the log-odds, so a linear model captures most of the signal and generalises without overfitting. |
| Decision Tree | Highest precision (0.668) but low recall (0.468): its predictions of churn are usually right, but it misses more than half of actual churners. Depth and leaf-size limits keep it from overfitting, and its AUC (0.837) is close to logistic regression, so its ranking is good but the default 0.5 threshold is conservative. |
| KNN | Middle-of-the-table on every metric (MCC 0.437). With 45+ mostly binary one-hot dimensions, distances become less discriminative, and k=25 smooths predictions. It is also the slowest at prediction time and the largest saved model (stores the full training set). |
| Naive Bayes | Lowest accuracy (0.696) and precision (0.460) but by far the highest recall (0.840): it flags most churners at the cost of many false alarms. The feature-independence assumption is violated (e.g. TotalCharges ≈ tenure × MonthlyCharges; streaming/security add-ons depend on InternetService), which double-counts evidence and pushes probabilities to extremes. Useful only if missing a churner is much costlier than a wasted retention offer. |
| Support Vector Machine | Accuracy (0.790) is competitive but AUC is the lowest (0.793) and recall is only 0.484. With the default C=1 and no class weighting, the RBF boundary favours the majority class; its margin scores rank customers less well than the probabilistic models. Class weighting or tuning C/gamma would likely help. |
| **Overall winner** | **Logistic Regression** — best Accuracy, AUC, F1 and MCC, plus it is fast, small and interpretable (coefficients show which factors drive churn). If the business priority is catching as many churners as possible, Naive Bayes (recall 0.84) or a lower decision threshold on Logistic Regression are alternatives. |

## e. Streamlit App Features

- CSV upload of test data (sidebar), plus a button to download the sample `test_data.csv`
- Model selection dropdown (5 models)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix heatmap, ROC curve and classification report
- "Compare all models" tab with a highlighted metrics table and bar chart
