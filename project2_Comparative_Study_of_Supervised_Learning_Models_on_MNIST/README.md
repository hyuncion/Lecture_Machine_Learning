# Project 2: Supervised Learning Model Comparison on MNIST

본 프로젝트는 **기계학습개론(Introduction to Machine Learning)** 수업의 두 번째 과제로,  
**MNIST handwritten digit dataset**을 대상으로 다양한 **supervised learning classification models**를 학습시키고,  
각 모델의 **주요 hyperparameter**가 분류 성능에 미치는 영향을 비교·분석하는 것을 목표로 합니다.

---

## Project Overview

MNIST는 0부터 9까지의 손글씨 숫자 이미지로 구성된 대표적인 classification benchmark dataset입니다.  
본 프로젝트에서는 동일한 데이터 전처리 파이프라인 하에서 여러 지도학습 모델을 적용하여  
**model characteristics**, **hyperparameter sensitivity**, **generalization performance**를 체계적으로 비교합니다.

---

## Models and Experiments

각 모델은 독립적인 Python 파일로 구현되었으며,  
하이퍼파라미터 변화에 따른 **test accuracy**를 기준으로 best model을 도출합니다.

---

### 1. Logistic Regression
**File:** `logistic_regression_mnist.py`

- Solver 비교: `lbfgs`, `newton-cg`, `sag`, `saga`
- `max_iter` 변화에 따른 수렴 및 성능 비교
- Regularization strength (`C`) 변화 분석

**Focus**
- Optimization method(solver)의 영향
- Regularization과 generalization의 관계

---

### 2. K-Nearest Neighbors (KNN)
**File:** `knn_mnist.py`

- `n_neighbors` 변화 (3, 5, 7, 9)
- Weighting strategy 비교 (`uniform` vs. `distance`)

**Focus**
- Model complexity와 neighborhood size의 관계
- Distance-based weighting 효과

---

### 3. Support Vector Machine (SVM)
**File:** `svm_mnist.py`

- Regularization parameter `C` 변화
- Kernel 비교: `linear`, `rbf`, `poly`, `sigmoid`

**Focus**
- Margin control과 overfitting
- Non-linear decision boundary의 효과

---

### 4. Random Forest
**File:** `random_forest_mnist.py`

- Number of trees (`n_estimators`) 변화
- Ensemble size에 따른 성능 변화 분석

**Focus**
- Ensemble learning의 효과
- Bias–variance trade-off 관점에서의 tree 개수

---

### 5. Gradient Boosting
**File:** `gradient_boosting_mnist.py`

- `n_estimators` 변화
- `learning_rate` 변화 분석

**Focus**
- Boosting 방식의 sequential learning 특성
- Learning rate와 overfitting의 관계

---

## Common Pipeline

모든 모델은 동일한 전처리 과정을 거칩니다:

1. MNIST dataset 로드 (`fetch_openml`)
2. Train / Test split
3. Feature scaling (`StandardScaler`)
4. Model training
5. Test accuracy evaluation

이를 통해 모델 간 **공정한 성능 비교**가 가능하도록 구성하였습니다.

---

## Implementation Details

- **Language:** Python
- **Libraries**
  - scikit-learn
  - NumPy
- **Dataset:** MNIST (`mnist_784`)
- Test accuracy를 기준으로 성능 평가

---

## Key Learning Outcomes

- 다양한 supervised learning 모델의 특성 이해
- Hyperparameter tuning이 성능에 미치는 영향 분석
- Linear vs. non-linear model 비교
- Ensemble methods(Random Forest, Gradient Boosting)의 장단점 이해
- 동일한 데이터에서 모델 선택 기준에 대한 실험적 근거 확보

---

## Notes

- 본 프로젝트는 **educational purpose**로 구현되었습니다.
- Computational cost를 고려하여 기본적인 train/test split을 사용하였습니다.
- Random seed 및 데이터 분할에 따라 결과는 다소 달라질 수 있습니다.

---

본 프로젝트는 이후 **unsupervised learning**, **dimensionality reduction**,  
그리고 **recommendation system** 프로젝트로 확장되는 기계학습 실험의 기반이 되는 비교 분석 과제입니다.
