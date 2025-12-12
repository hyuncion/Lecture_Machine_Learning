# Project 3: Unsupervised Learning and Dimensionality Reduction on MNIST

본 프로젝트는 **기계학습개론(Introduction to Machine Learning)** 수업의 세 번째 과제로,  
**MNIST handwritten digit dataset**을 대상으로 다양한 **unsupervised learning (clustering)** 기법을 적용하고,  
정량적 평가 지표와 차원 축소 기법을 통해 데이터의 구조적 특성을 분석하는 것을 목표로 합니다.

---

## Project Overview

Supervised learning과 달리 unsupervised learning에서는 정답 label 없이 데이터의 잠재 구조를 파악해야 합니다.  
본 프로젝트에서는 MNIST 데이터에 대해 여러 **clustering algorithms**를 적용하고,  
**Adjusted Rand Index**와 **Adjusted Mutual Information**을 사용해 clustering 결과를 정량적으로 평가합니다.  
또한 **PCA**와 **Kernel PCA**를 활용하여 차원 축소가 clustering 성능에 미치는 영향을 분석합니다.

---

## Clustering Algorithm Comparison

먼저, 실제 데이터에 적용하기에 앞서 **synthetic datasets**를 사용하여 다양한 clustering 알고리즘의 특성을 비교합니다.

- 사용한 데이터셋: circles, moons, blobs, anisotropic data 등
- 비교한 알고리즘:
  - MiniBatch K-Means
  - Affinity Propagation
  - Mean Shift
  - Spectral Clustering
  - Agglomerative Clustering (Ward 포함)
  - DBSCAN
  - OPTICS
  - BIRCH
  - Gaussian Mixture Model (GMM)

각 알고리즘의 clustering 결과와 실행 시간을 시각화하여,  
데이터 분포에 따른 알고리즘의 강점과 한계를 분석합니다.

---

## MNIST Clustering and Evaluation

MNIST 데이터셋에서 각 digit class당 100개씩 샘플링하여 총 **1,000개의 이미지**를 구성한 뒤,  
다음 clustering 알고리즘을 적용합니다 (k = 10):

- Agglomerative Clustering
- K-Means
- Gaussian Mixture Model
- Spectral Clustering

Clustering 결과는 다음 지표로 평가합니다:

- **Adjusted Rand Index (ARI)**
- **Adjusted Mutual Information (AMI)**

이를 통해 clustering 결과와 실제 label 간의 일치도를 비교하고,  
각 알고리즘의 상대적인 성능을 분석합니다.

---

## Clustering + 1-NN Classification

Clustering 결과로부터 각 cluster의 중심을 추출한 뒤,  
이를 기반으로 **1-Nearest Neighbor (1-NN)** classifier를 구성하여 MNIST 데이터를 분류합니다.

이 과정은:
- Clustering 결과가 분류 관점에서 얼마나 의미 있는 구조를 형성하는지
- Unsupervised learning과 supervised learning의 연결 가능성

을 실험적으로 확인하기 위한 단계입니다.

---

## Dimensionality Reduction: PCA and Kernel PCA

Clustering에 사용된 MNIST 샘플에 대해 **PCA**와 **Kernel PCA**를 적용하여:

- Mean image 시각화
- 상위 eigenvectors (주성분) 시각화
- Eigenvalues 분포 분석

을 수행합니다.

또한 차원 축소된 feature space에서 **K-Means clustering**을 다시 수행하고,  
ARI 및 AMI를 계산하여 차원 축소가 clustering 성능에 미치는 영향을 분석합니다.

---

## Visualization and Error Analysis

마지막으로 **1-NN classifier**를 사용해 MNIST test set을 분류하고:

- 각 digit class별 correctly classified images
- incorrectly classified images

를 시각화하여 모델의 한계와 오류 패턴을 분석합니다.

---

## Implementation Details

- **Language:** Python
- **Libraries**
  - NumPy
  - Matplotlib
  - scikit-learn
- **Dataset:** MNIST (`mnist_784`)
- Evaluation metrics: Adjusted Rand Index, Adjusted Mutual Information

---

## Key Learning Outcomes

- 다양한 clustering 알고리즘의 가정과 특성 이해
- Unsupervised learning 결과를 정량적으로 평가하는 방법
- Clustering과 classification 간의 관계 이해
- PCA / Kernel PCA의 역할과 한계
- High-dimensional data에서 차원 축소 선택의 중요성
- 시각화를 통한 error analysis 능력 향상

---

## Notes

- 본 프로젝트는 **educational purpose**로 구현되었습니다.
- MNIST는 고차원 데이터이므로 clustering 성능에는 한계가 존재합니다.
- 차원 축소 기법 및 파라미터 선택에 따라 결과는 달라질 수 있습니다.

---

본 프로젝트는 이후 **recommendation system** 및 **representation learning**으로 확장되는  
unsupervised learning의 기초 개념을 실험적으로 이해하는 데 중점을 둡니다.
