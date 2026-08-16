# PythonProject-MLAssignment2
Respository for  Machine Learning Assignment
# Machine Learning Assignment 2: Classification Models & Deployment

## a. Problem Statement
The objective of this project is to build, evaluate, and compare 5 distinct Machine Learning classification algorithms on a tabular dataset with at least 12 features and 500 instances, and deploy an interactive evaluation application using Streamlit Community Cloud.

## b. Dataset Description
- **Instances:** 1,000
- **Features:** 14 numeric features
- **Target Variable:** Binary classification label (0 or 1)
- **Source:** Custom synthetic classification dataset meeting BITS WILP criteria (>= 12 features, >= 500 rows).

## c. GitHub Repository Link
[Insert Your Repository Link Here]

## d. Models & Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8450 | 0.9120 | 0.8460 | 0.8450 | 0.8448 | 0.6902 |
| Decision Tree | 0.8100 | 0.8100 | 0.8105 | 0.8100 | 0.8101 | 0.6201 |
| kNN | 0.8300 | 0.8850 | 0.8320 | 0.8300 | 0.8298 | 0.6605 |
| Naive Bayes | 0.8250 | 0.8950 | 0.8270 | 0.8250 | 0.8247 | 0.6508 |
| Random Forest (Ensemble) | **0.8850** | **0.9510** | **0.8860** | **0.8850** | **0.8849** | **0.7704** |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Provides a strong linear baseline with smooth decision boundaries and fast inference. |
| **Decision Tree** | Captures non-linear feature interactions well but exhibits slight variance and susceptibility to overfitting. |
| **kNN** | Performs decently after feature scaling, sensitive to local neighborhood density. |
| **Naive Bayes** | Executes extremely fast; conditional independence assumption holds reasonably well across continuous features. |
| **Random Forest (Ensemble)** | Outperforms all individual models across Accuracy, AUC, and MCC due to bagging and variance reduction. |
| **Overall Winner** | **Random Forest (Ensemble)** achieves the highest accuracy (88.5%) and MCC score (0.7704). |
