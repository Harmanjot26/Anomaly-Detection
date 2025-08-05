# 🔍 Anomaly Detection in Credit Card Transactions

This project implements an **unsupervised anomaly detection system** to identify fraudulent transactions in a real-world, highly imbalanced credit card dataset. The aim is to distinguish rare and unusual behaviors from typical patterns of legitimate spending using machine learning and statistical techniques.

## 📌 Problem Statement

Credit card fraud is a persistent threat in financial systems. In this project:
- We treat fraudulent records as anomalies.
- We build a data-driven pipeline for **fraud detection**.
- The system flags suspicious transactions that deviate from authentic behavior for further investigation.

## 📊 Dataset Overview

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Records**: 284,807 transactions
- **Fraudulent Records**: 492 (0.17%)
- **Features**:
  - `Time`, `Amount`, `Class` (target)
  - `V1` to `V28`: PCA-transformed anonymized features

## 🧠 Methodology

1. **Preprocessing & Feature Selection**
   - Analyzed correlation heatmaps.
   - Selected most relevant features to reduce dimensionality and noise.

2. **Data Transformation**
   - Normalization of `Amount` and `Time`.
   - Visual exploration of distributions.

3. **Modeling Approaches**
   - Used **Isolation Forest** and **Local Outlier Factor** for unsupervised detection.
   - Evaluated models using:
     - Precision, Recall, F1-score
     - Confusion matrix
     - ROC-AUC curve

4. **Evaluation Strategy**
   - Metrics were selected to reduce false positives while maximizing fraud detection.
   - Visual diagnostics supported anomaly confirmation.

## 📈 Visualizations

- Correlation heatmap for feature insight
- Class distribution histogram
- Anomaly scatter plots with detected fraudulent points
- Confusion matrix and performance metrics visualization

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**:
  - `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`
- **Notebook Tools**: Jupyter Notebook

## 📎 Sample Code Snippet

```python
from sklearn.ensemble import IsolationForest
model = IsolationForest(n_estimators=100, contamination=0.002, random_state=42)
model.fit(X_train)
y_pred = model.predict(X_test)


