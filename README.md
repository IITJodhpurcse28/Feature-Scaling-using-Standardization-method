# Feature Scaling using Standardization

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-red.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple.svg)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

A machine learning preprocessing project demonstrating the effect of **Standardization (Z-Score Scaling)** on data and model performance.
---

## Overview

Feature scaling is an important preprocessing step in machine learning. When features have different ranges, some algorithms may give more importance to larger-valued features.

This project demonstrates:

- Loading and preprocessing data
- Train-Test Split
- Applying Standardization
- Visualizing data before and after scaling
- Training Logistic Regression
- Comparing model performance before and after scaling

---

## Dataset

Dataset Used: **Social Network Ads Dataset**

Features:

- Age
- EstimatedSalary

Target:

- Purchased

---

## Standardization Formula

Standardization transforms data so that it has:

- Mean = 0
- Standard Deviation = 1

Formula:

\[
z = \frac{x - \mu}{\sigma}
\]

Where:

- \(x\) = Original value
- \(\mu\) = Mean of feature
- \(\sigma\) = Standard deviation of feature
- \(z\) = Standardized value

---

## Why Standardization?

Consider:

| Feature | Range |
|----------|--------|
| Age | 18 - 60 |
| Salary | 15,000 - 150,000 |

Salary values are much larger than Age values.

Without scaling, many ML algorithms may treat Salary as more important simply because of its larger magnitude.

Standardization puts all features on a similar scale.

---

## When Should We Use Standardization?

Use Standardization when working with:

### Distance-Based Algorithms

- KNN
- K-Means Clustering
- DBSCAN

### Gradient-Based Algorithms

- Logistic Regression
- Linear Regression (Gradient Descent)
- Neural Networks

### Margin-Based Algorithms

- SVM (Support Vector Machine)

### Dimensionality Reduction

- PCA

### Regularized Models

- Ridge Regression
- Lasso Regression
- Elastic Net

---

## When Scaling is Usually Not Required?

Tree-based algorithms generally do not require feature scaling:

- Decision Trees
- Random Forest
- XGBoost
- LightGBM
- CatBoost

These models split data based on feature values rather than distances.

---

## Implementation Steps

### 1. Load Dataset

```python
df = pd.read_csv('Social_Network_Ads.csv')
```

### 2. Train-Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(
    df.drop('Purchased', axis=1),
    df['Purchased'],
    test_size=0.3,
    random_state=0
)
```

### 3. Apply Standardization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

### 4. Train Logistic Regression

```python
lr = LogisticRegression()
lr_scaled = LogisticRegression()

lr.fit(x_train, y_train)
lr_scaled.fit(x_train_scaled, y_train)
```

### 5. Compare Accuracy

```python
accuracy_score(y_test, y_pred)
accuracy_score(y_test, y_pred_scaled)
```

---

## Visualization

### Before Scaling

- Features have different ranges.
- Salary dominates the scale.

### After Scaling

- Mean becomes approximately 0.
- Standard deviation becomes approximately 1.
- Features become comparable.

---

## Results

The notebook compares Logistic Regression performance before and after standardization.

In many cases, scaling improves:

- Model convergence
- Training stability
- Prediction accuracy

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

## Author

Rahul Saxena

- GATE CSIT 2026 AIR 1129
- GATE CSIT 2025 AIR 1852
- Competitive Programmer
- Machine Learning Enthusiast

---
