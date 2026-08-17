# 🏦 Automated Loan Approval & Risk Intelligence Platform

An end-to-end Machine Learning web application and analytics workbench built for predicting loan application approvals and assessing financial risk. Developed as part of the **BITS Pilani — Machine Learning Assignment 2** curriculum.

---

## 📌 Problem Statement

Financial institutions evaluate loan applications based on applicant creditworthiness and financial profiles. The objective of this project is to build, evaluate, and compare multiple supervised machine learning classifiers to automate loan eligibility predictions while minimizing credit risk and maximizing predictive accuracy.

---

## 📊 Dataset Description

* **Data Source:** [Kaggle Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
* **Primary Dataset:** `train.csv` (614 instances, 16 features)
* **Target Variable:** `Loan_Status` (`Y` = Approved / `1`, `N` = Rejected / `0`)
* **Input Features:**
  * **Numerical (7):** `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Existing_Loans`, `Age`
  * **Categorical (6):** `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`

---

## 🔗 Repository & Application Links

* **GitHub Repository Link:** https://github.com/2025ac05389/PythonProject-MLAssignment2
* **Live Streamlit App:** https://pythonproject-mlassignment2-tauheed.streamlit.app

---

## 🤖 Models Used & Comparison Table

The following 5 classification models were trained and benchmarked across all 6 mandatory evaluation metrics:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8537 | 0.8505 | 0.8384 | 0.9765 | 0.9022 | 0.6476 |
| **Decision Tree** | 0.8537 | 0.7523 | 0.8317 | 0.9882 | 0.9032 | 0.6521 |
| **kNN** | 0.8374 | 0.8393 | 0.8495 | 0.9294 | 0.8876 | 0.6036 |
| **Naive Bayes** | 0.8455 | 0.7892 | 0.8367 | 0.9647 | 0.8962 | 0.6242 |
| **Random Forest (Ensemble)** | 0.7886 | 0.7834 | 0.8242 | 0.8824 | 0.8523 | 0.4858 |

---

## 📝 Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Strongest overall performance because it achieved the highest average score across the six required evaluation metrics. |
| **Decision Tree** | Accuracy=0.8537, AUC=0.7523, Precision=0.8317, Recall=0.9882, F1=0.9032, MCC=0.6521. |
| **kNN** | Accuracy=0.8374, AUC=0.8393, Precision=0.8495, Recall=0.9294, F1=0.8876, MCC=0.6036. |
| **Naive Bayes** | Accuracy=0.8455, AUC=0.7892, Precision=0.8367, Recall=0.9647, F1=0.8962, MCC=0.6242. |
| **Random Forest (Ensemble)** | Accuracy=0.7886, AUC=0.7834, Precision=0.8242, Recall=0.8824, F1=0.8523, MCC=0.4858. |
| **Overall Winner for your dataset?** | **Logistic Regression** (achieved the highest average across all six evaluation metrics = 0.8448). |

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
├── train.csv                   # Full baseline training dataset (614 rows)
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
