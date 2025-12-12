# Project 1: Polynomial Regression and Regularization Analysis

본 프로젝트는 **기계학습개론(Introduction to Machine Learning)** 수업의 첫 번째 과제로,  
**Polynomial Regression**을 중심으로 **overfitting**, **bias–variance trade-off**, 그리고  
**regularization (L1 / L2)** 및 **training data size**가 모델의 일반화 성능에 미치는 영향을 실험적으로 분석하는 것을 목표로 합니다.

---

## Project Overview

타겟 함수  

$$ y = \sin(2\pi x) $$

를 근사하는 regression model을 구성하고, 모델의 **complexity**를 점진적으로 증가시키며 학습 결과를 비교합니다.  
또한 **outlier 추가**, **regularization 적용**, **training sample 증가**를 통해 모델의 거동 변화를 관찰합니다.

---

## Experiments

### 1. Noisy Data Generation
- Uniform distribution \([0,1]\)에서 **10 samples** 생성
- Gaussian noise 추가
- Ground truth function과 sampled data 시각화

**Goal:**  
적은 데이터와 noise가 regression model에 미치는 영향 이해

---

### 2. Polynomial Regression with Increasing Degree
- Polynomial degrees: **1, 3, 5, 9, 15**
- Ordinary Least Squares 기반 regression 수행
- 각 degree별 prediction curve 비교

**Observation**
- Degree 증가에 따라 model complexity 증가
- High-degree polynomial에서 명확한 overfitting 발생

---

### 3. Effect of Outliers
- Target function을 따르지 않는 **2–3 outliers** 추가
- 동일한 polynomial regression 실험 반복

**Observation**
- Outliers로 인해 model이 ground truth에서 크게 벗어남
- High-degree model일수록 overfitting이 더욱 심화됨

---

### 4. Regularization: Ridge vs. Lasso
- **Ridge Regression (L2 regularization)**
- **Lasso Regression (L1 regularization)**
- Polynomial degree: **9, 15**
- Regularization strength \(\alpha\) 변화

**Comparison**
- **L2:** 전체 weight를 부드럽게 shrink
- **L1:** 일부 weight를 0으로 만들어 feature selection 효과
- 과도한 regularization은 underfitting 유발

---

### 5. Effect of Training Data Size
- Training samples: **10 → 100**
- 동일한 polynomial regression 실험 수행

**Observation**
- Sample 수 증가로 overfitting 완화
- High-degree model도 비교적 안정적으로 generalize
- Training data size의 중요성 확인

---

## Implementation Details

- **Language:** Python
- **Libraries**
  - NumPy
  - Matplotlib
  - scikit-learn  
    (`PolynomialFeatures`, `LinearRegression`, `Ridge`, `Lasso`)
- Reproducibility를 위해 random seed 고정
- `main.py`에서 각 experiment를 함수 단위로 실행

---

## Key Learning Outcomes

- Polynomial degree와 model complexity의 관계
- Bias–variance trade-off에 대한 직관적 이해
- Outlier가 regression model에 미치는 영향
- L1 vs. L2 regularization의 차이점
- Training data size와 generalization 성능의 관계
- Visualization을 통한 model behavior 분석 능력

---

## Notes

- 본 프로젝트는 **educational purpose**로 구현되었습니다.
- Synthetic dataset을 사용하여 개념 분석에 집중하였습니다.

---

본 프로젝트는 이후 지도학습 모델과 정규화 기법을 이해하기 위한 기초 실험으로,  
machine learning 전반의 핵심 개념을 체계적으로 학습하는 데 목적이 있습니다.
