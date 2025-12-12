# 🎓 Machine Learning Course Projects

본 저장소는 **기계학습개론(Introduction to Machine Learning)** 강의를 수강하며 수행한  
총 **4개의 프로젝트**를 정리한 저장소입니다.  
각 프로젝트는 머신러닝의 핵심 주제를 단계적으로 다루며,  
이론적 개념을 **직접 구현하고 실험을 통해 검증**하는 데 초점을 두었습니다.

---

## 📌 Project Overview

### Project 1. Polynomial Regression and Regularization Analysis
**(Bias–Variance Trade-off, Overfitting, L1/L2 Regularization)**

- Synthetic data \(y = \sin(2\pi x)\)를 대상으로 다항 회귀 수행
- Polynomial degree 변화에 따른 overfitting 분석
- Outlier 추가 실험
- **Ridge(L2)** / **Lasso(L1)** 정규화 비교
- Training data size 증가 효과 분석

**Key focus**
- Bias–variance trade-off
- Model complexity와 generalization
- Regularization의 역할

---

### Project 2. Supervised Learning Model Comparison on MNIST
**(Classification & Hyperparameter Analysis)**

- MNIST handwritten digit dataset 사용
- 다양한 supervised learning 모델 비교:
  - Logistic Regression
  - K-NN
  - SVM
  - Random Forest
  - Gradient Boosting
- 각 모델의 핵심 hyperparameter 변화에 따른 성능 분석
- Test accuracy 기준 best model 도출

**Key focus**
- Model selection
- Hyperparameter sensitivity
- Linear vs. non-linear models
- Ensemble learning 이해

---

### Project 3. Unsupervised Learning and Dimensionality Reduction on MNIST
**(Clustering, Evaluation Metrics, PCA / Kernel PCA)**

- 다양한 clustering algorithm 비교 (toy dataset & MNIST)
- MNIST clustering 결과에 대한 정량 평가:
  - Adjusted Rand Index
  - Adjusted Mutual Information
- Clustering + **1-NN classification** 실험
- **PCA / Kernel PCA**를 통한 차원 축소 및 성능 영향 분석
- Error analysis 및 시각화

**Key focus**
- Unsupervised learning의 평가 방법
- High-dimensional data 분석
- Clustering과 classification의 연결

---

### Project 4. Collaborative Filtering
**(Recommender Systems: K-NN vs. Matrix Factorization)**

- MovieLens dataset 기반 추천 시스템 구현
- Similarity-based collaborative filtering:
  - Item–Item K-NN
  - 다양한 distance / similarity metric 비교
- Model-based collaborative filtering:
  - Singular Value Decomposition (SVD)
  - Alternating Least Squares (ALS)
- 추천 결과의 특성과 차이 분석

**Key focus**
- Recommender system의 핵심 접근법
- Similarity-based vs. latent factor models
- Sparsity와 scalability 이슈

---

## 🛠 Tools & Libraries

- Python
- NumPy
- Matplotlib
- scikit-learn
- implicit (ALS)
- Jupyter Notebook

---

## 🎯 Overall Learning Outcomes

이 저장소를 통해 다음과 같은 머신러닝 핵심 역량을 학습했습니다:

- 모델 복잡도와 일반화 성능 간의 관계 이해
- Supervised / Unsupervised learning 전반에 대한 실험적 이해
- 정량적 평가 지표의 중요성
- Hyperparameter tuning의 실제적 영향
- 고차원 데이터와 차원 축소 기법의 역할
- 추천 시스템의 기본 구조와 확장 가능성

---

## 📌 Notes

- 모든 프로젝트는 **educational purpose**로 구현되었습니다.
- 실험 결과는 데이터 분할, random seed, 하이퍼파라미터 설정에 따라 달라질 수 있습니다.
- 각 프로젝트는 독립적으로 실행 및 분석 가능하도록 구성되어 있습니다.

---

## ✨ Takeaway

본 저장소는  
> **“머신러닝의 핵심 개념을 직접 구현하고, 실험을 통해 이해한다”**  
라는 목표 아래 진행된 프로젝트들의 집합입니다.

기초적인 회귀 문제부터 분류, 클러스터링, 추천 시스템까지  
머신러닝의 주요 문제 유형을 단계적으로 다루며,  
이론과 구현을 연결하는 경험을 담고 있습니다.
