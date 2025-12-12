# Project 1: Polynomial Regression and Regularization Analysis

This project is the first assignment of the **Introduction to Machine Learning** course.  
The goal is to understand **polynomial regression**, **overfitting**, and the **bias–variance trade-off** through controlled experiments on synthetic data, and to analyze how **regularization** and **training data size** affect model behavior.

---

## Overview

We model a target function  
\[
y = \sin(2\pi x)
\]
using polynomial regression under different conditions.  
By progressively increasing model complexity, adding outliers, applying regularization, and increasing the number of training samples, we empirically study fundamental machine learning concepts.

---

## Experiments

### 1. Noisy Data Generation
- Generate **10 samples** uniformly in \([0,1]\)
- Add Gaussian noise to the target function
- Visualize sampled points and the true function

**Purpose:**  
Understand the effect of noise and limited data on regression.

---

### 2. Polynomial Regression with Increasing Degree
- Polynomial degrees: **1, 3, 5, 9, 15**
- Train models using ordinary least squares
- Visualize fitted curves

**Observation:**
- Higher-degree polynomials fit training data more closely
- Severe **overfitting** appears for large degrees

---

### 3. Effect of Outliers
- Add **2–3 artificial outliers**
- Re-train polynomial regression models

**Observation:**
- Outliers significantly distort high-degree models
- Overfitting becomes more severe with noisy and corrupted data

---

### 4. Regularization: Ridge vs. Lasso
- Apply **Ridge (L2)** and **Lasso (L1)** regression
- Polynomial degrees: **9, 15**
- Vary regularization strength \(\alpha\)

**Key comparisons:**
- **Ridge (L2):** Smoothly suppresses large weights
- **Lasso (L1):** Can drive some coefficients to zero (feature selection)
- Excessive regularization leads to underfitting

---

### 5. Increasing Training Data Size
- Increase samples from **10 → 100**
- Repeat polynomial regression experiments

**Observation:**
- More data significantly reduces overfitting
- High-degree models generalize better with sufficient samples
- Demonstrates the importance of training data size

---

## Implementation Details

- **Language:** Python
- **Libraries:**
  - NumPy
  - Matplotlib
  - scikit-learn (`LinearRegression`, `Ridge`, `Lasso`, `PolynomialFeatures`)
- Random seed fixed for reproducibility
- A single `main.py` file is used to execute each experiment independently

---

## Key Learning Outcomes

- Practical understanding of **bias–variance trade-off**
- How polynomial degree affects model complexity
- Sensitivity of regression models to **outliers**
- Role of **L1 vs. L2 regularization**
- Importance of **training data size** in generalization
- Empirical intuition behind overfitting and underfitting

---

## Notes

- This project uses **synthetic data** for controlled analysis
- Implemented for **educational purposes**
- Visualizations are essential for interpreting model behavior

---

This project builds a strong foundation for understanding supervised learning models and prepares for more complex algorithms studied in later projects.
