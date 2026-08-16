import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# ==============================================================================
# 1. Project paths
#    Relative paths are required for Streamlit Community Cloud deployment.
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "")
DATASET_PATH = os.path.join(BASE_DIR, "../train.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================================================================
# 2. Load dataset
# ==============================================================================

df = pd.read_csv(DATASET_PATH)

if 'Loan_ID' in df.columns:
    df = df.drop(columns=['Loan_ID'])

# Basic assignment validation
print(f"Dataset shape: {df.shape}")

if len(df) < 500:
    print("WARNING: Assignment requires at least 500 instances.")

feature_columns = [c for c in df.columns if c != 'Loan_Status']
print(f"Number of input features: {len(feature_columns)}")

if len(feature_columns) < 12:
    print(
        "WARNING: Assignment requires a minimum of 12 features. "
        f"Current dataset has {len(feature_columns)} input features."
    )

# ==============================================================================
# 3. Separate features and target
# ==============================================================================

X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

# ==============================================================================
# 4. Define numerical and categorical features
# ==============================================================================

num_cols = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History',
    'Existing_Loans',
    'Age'
]

cat_cols = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Property_Area'
]

# Keep only columns actually present in the selected dataset.
num_cols = [c for c in num_cols if c in X.columns]
cat_cols = [c for c in cat_cols if c in X.columns]

# ==============================================================================
# 5. Build preprocessing pipelines
# ==============================================================================

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore'
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)

# ==============================================================================
# 6. Stratified Train/Test Split (80/20)
# ==============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================================================================
# 7. Export test_data.csv for Streamlit UI
# ==============================================================================

test_df = X_test.copy()
test_df['Loan_Status'] = y_test.map({1: 'Y', 0: 'N'})
test_df.to_csv(os.path.join(BASE_DIR, '../test_data.csv'), index=False)

print("✓ Exported test_data.csv successfully!")

# ==============================================================================
# 8. Fit preprocessing pipeline and transform features
# ==============================================================================

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.joblib'))
print("✓ Saved preprocessor.joblib")

# ==============================================================================
# 9. Instantiate the 5 classification models required by the assignment
# ==============================================================================

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ),
    'kNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest (Ensemble)': RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# ==============================================================================
# 10. Train, evaluate, and save models
# ==============================================================================

results = []

for name, model in models.items():

    model.fit(X_train_prep, y_train)

    # Save trained model
    file_name = (
        name.lower()
        .replace(' ', '_')
        .replace('_(ensemble)', '')
        + '.joblib'
    )

    joblib.dump(
        model,
        os.path.join(MODEL_DIR, file_name)
    )

    # Test predictions
    y_pred = model.predict(X_test_prep)

    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_prep)[:, 1]
    else:
        y_proba = model.decision_function(X_test_prep)

    # Required six metrics
    results.append({
        'ML Model Name': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC': round(roc_auc_score(y_test, y_proba), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    })

# ==============================================================================
# 11. Comparison table
# ==============================================================================

results_df = pd.DataFrame(results)

print("\n=== FINAL EVALUATION METRICS COMPARISON TABLE ===")
print(results_df.to_string(index=False))

# ==============================================================================
# 12. Determine overall winner
#     The assignment asks for an overall winner. We use the average of the
#     six required metrics so that all required metrics contribute equally.
# ==============================================================================

metric_columns = [
    'Accuracy',
    'AUC',
    'Precision',
    'Recall',
    'F1',
    'MCC'
]

results_df['Overall Score'] = results_df[metric_columns].mean(axis=1)

winner_row = results_df.loc[
    results_df['Overall Score'].idxmax()
]

overall_winner = winner_row['ML Model Name']

print("\n=== OVERALL WINNER ===")
print(f"{overall_winner} "
      f"(average of six metrics = {winner_row['Overall Score']:.4f})")

# ==============================================================================
# 13. Generate concise observations for README / report
# ==============================================================================

print("\n=== MODEL OBSERVATIONS ===")

obs_list = []

for _, row in results_df.iterrows():

    model_name = row['ML Model Name']

    if model_name == overall_winner:
        observation = (
            "Strongest overall performance because it achieved the highest "
            "average score across the six required evaluation metrics."
        )
    else:
        observation = (
            f"Accuracy={row['Accuracy']:.4f}, AUC={row['AUC']:.4f}, "
            f"Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, "
            f"F1={row['F1']:.4f}, MCC={row['MCC']:.4f}."
        )

    print(f"{model_name}: {observation}")
    obs_list.append(f"* **{model_name}:** {observation}")

# Format observations for Markdown insertion
model_observations_text = "\n".join(obs_list)

# Generate README.md content dynamically
readme_content = f"""# 🏦 Automated Loan Approval & Risk Intelligence Platform

An end-to-end Machine Learning web application and analytics workbench built for predicting loan application approvals and assessing financial risk. Developed as part of the **BITS Pilani — Machine Learning Assignment 2** curriculum.

---

## 📌 Project Overview

This repository features a complete machine learning workflow for binary classification on loan approval data:
1. **Data Preprocessing & Feature Engineering:** Handles missing values, scales numerical features, and encodes categorical attributes.
2. **Multi-Model Training Pipeline:** Trains and evaluates **5 standard machine learning classifiers**.
3. **Interactive Streamlit Dashboard:** Provides an interactive web interface for real-time model evaluation, threshold tuning, batch inference, and risk profiling.
4. **GitHub Repository:** https://github.com/2025ac05389/PythonProject-MLAssignment2
5. **StreamlitAPP:** https://pythonproject-mlassignment2-tauheed.streamlit.app
---

## 📊 Dataset & Features
* **Data Source:** https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
* **Primary Dataset:** `train.csv` ({len(df)} instances, {len(feature_columns)} features)
* **Target Variable:** `Loan_Status` (`Y` = Approved / `1`, `N` = Rejected / `0`)
* **Input Features:**
  * **Numerical ({len(num_cols)}):** {', '.join([f'`{c}`' for c in num_cols])}
  * **Categorical ({len(cat_cols)}):** {', '.join([f'`{c}`' for c in cat_cols])}

---

## 🤖 Machine Learning Classifiers

The training pipeline (`train_model.py`) trains and evaluates the following 5 models:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **k-Nearest Neighbors (kNN)**
4. **Naive Bayes (GaussianNB)**
5. **Random Forest Classifier (Ensemble)**

### 📈 Evaluation Metrics
Every model is benchmarked across **6 mandatory evaluation metrics**:
* **Accuracy**
* **Area Under the ROC Curve (AUC)**
* **Precision**
* **Recall**
* **F1 Score**
* **Matthews Correlation Coefficient (MCC)**

---

## 📝 Model Observations

{model_observations_text}

---

## 🚀 Interactive Streamlit Web Application (`app.py`)

The Streamlit UI provides rich evaluation and inference capabilities:
* **Dynamic Decision Cutoff:** Interactive slider (30%–90%) to adjust approval confidence thresholds in real time.
* **Seaborn Confusion Matrix Heatmap:** Displays clear visual counts for Actual vs. Predicted classifications.
* **ROC Curve Plotting:** Renders True Positive Rate vs. False Positive Rate curves with AUC indicators.
* **All-Model Benchmark Comparison Table:** Displays performance across all 5 algorithms and declares the top-performing model.
* **Batch Scoring & Export:** Predicts loan decisions for uploaded CSV datasets and provides a downloadable `loan_prediction_results.csv` file.

---

## 📁 Repository Structure

```text
.
├── train_model.py              # Script to train models, generate metrics, & export test_data.csv
├── app.py                      # Streamlit interactive application script
├── train.csv                   # Full baseline training dataset ({len(df)} rows)
├── test_data.csv               # Holdout evaluation test dataset generated after training
├── requirements.txt            # Python package dependencies
├── model/                      # Serialized model artifacts directory (.joblib)
│   ├── preprocessor.joblib     # Fitted ColumnTransformer pipeline
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── random_forest.joblib
│   └── model_metrics.csv       # Summary evaluation metrics table
└── README.md                   # Project documentation
"""

# 3. Write/Overwrite the README.md file
readme_path = os.path.join(BASE_DIR, "../README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

print(f"✓ Generated README.md successfully at {readme_path}")

# Save metrics so the Streamlit app can display all models.
results_df.to_csv(
    os.path.join(MODEL_DIR, 'model_metrics.csv'),
    index=False
)

print("\n✓ Saved model_metrics.csv")
print("✓ Training completed successfully.")