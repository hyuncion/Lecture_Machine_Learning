# Project 4: Collaborative Filtering

본 프로젝트는 **기계학습개론(Introduction to Machine Learning)** 수업의 네 번째 과제로,  
**Recommender System**의 핵심 개념인 **Collaborative Filtering**을 중심으로  
**similarity-based 방법(K-NN)**과 **latent factor 기반 방법(Matrix Factorization)**을 비교·분석합니다.

---

## Project Overview

추천 시스템은 사용자–아이템 상호작용 데이터를 기반으로 사용자의 선호를 예측하는 문제입니다.  
본 프로젝트에서는 MovieLens 데이터셋을 활용하여 다음 두 계열의 접근법을 구현하고 결과 차이를 분석합니다.

1. **Similarity-based Collaborative Filtering**
   - Item–Item K-NN
   - 다양한 distance / similarity metric 비교
2. **Model-based Collaborative Filtering**
   - Singular Value Decomposition (SVD)
   - Alternating Least Squares (ALS)

각 방법이 생성하는 추천 결과의 성격 차이와 장단점을 실험적으로 비교하는 것이 핵심 목표입니다.

---

## Similarity-based Collaborative Filtering (K-NN)

### Item–Item K-NN
- 각 아이템을 벡터로 표현
- 특정 아이템과 가장 유사한 아이템을 K-NN으로 탐색
- `NearestNeighbors`를 사용한 brute-force 검색

### Distance / Similarity Metrics
다음 metric에 따라 추천 결과가 어떻게 달라지는지 비교합니다:
- **Cosine similarity**
- **Euclidean (L2)**
- **Manhattan / Cityblock (L1)**
- **Minkowski**

**Observation**
- Metric 선택에 따라 이웃 아이템의 성격이 달라짐
- Cosine similarity는 벡터 크기보다 방향에 집중
- L1 / L2 계열은 절대적인 거리 차이에 민감

---

## Matrix Factorization-based Collaborative Filtering

### 1. Singular Value Decomposition (SVD)
- 사용자–아이템 행렬을 저차원의 latent space로 분해
- Truncated SVD를 사용하여 계산 복잡도 완화
- Latent factor 기반으로 아이템 간 유사도 계산

**특징**
- Sparse matrix 문제 완화
- K-NN과는 다른 추천 패턴 생성
- Cold start 문제에는 취약

---

### 2. Alternating Least Squares (ALS)
- 사용자 latent factor와 아이템 latent factor를 번갈아 최적화
- Sparse matrix를 직접 고려하는 방식
- 대규모 데이터셋에 적합

**특징**
- SVD와 유사하지만 학습 방식이 다름
- Scalability 우수
- Hyperparameter 설정에 따라 결과 민감

---

## Comparison and Analysis

| 방법 | 장점 | 한계 |
|----|----|----|
| K-NN | 직관적, 해석 용이 | 대규모 데이터에 비효율적 |
| SVD | Latent representation 학습 | Cold start 문제 |
| ALS | Scalability 우수, sparse 대응 | 파라미터 민감 |

**핵심 비교 포인트**
- Local similarity (K-NN) vs. global structure (Matrix Factorization)
- Distance metric의 영향
- 추천 결과의 다양성과 안정성

---

## Implementation Details

- **Language:** Python
- **Libraries**
  - NumPy
  - scikit-learn
  - implicit (ALS)
- **Dataset:** MovieLens
- 분석 및 시각화는 `total_analysis.ipynb`에서 수행

---

## Key Learning Outcomes

- Collaborative filtering의 두 가지 핵심 접근 방식 이해
- Similarity-based vs. model-based recommendation 비교
- Distance metric 선택의 중요성
- Latent factor 모델의 개념적 이해
- Recommender system에서 scalability와 sparsity 이슈 체감

---

## Notes

- 본 프로젝트는 **educational purpose**로 구현되었습니다.
- 데이터 전처리 및 하이퍼파라미터에 따라 결과는 달라질 수 있습니다.
- 실제 서비스 환경에서는 hybrid 방식이 주로 사용됩니다.

---

본 프로젝트는 추천 시스템의 기초 개념을 실험적으로 이해하고,  
이후 **advanced recommender system** 및 **representation learning**으로 확장하기 위한 기반을 제공합니다.
