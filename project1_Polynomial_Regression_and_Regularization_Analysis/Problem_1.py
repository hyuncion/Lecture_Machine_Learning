import numpy as np
import matplotlib.pyplot as plt

def sin2pi(X):
    return np.sin(2 * np.pi * X)

def Problem1():
    # 데이터 생성
    np.random.seed(1)
    n_samples = 10
    X = np.sort(np.random.rand(n_samples))
    y = np.sin(2 * np.pi * X) + np.random.randn(n_samples) * 0.1

    # 시각화
    plt.scatter(X, y)
    plt.plot(np.linspace(0, 1, 100), sin2pi(np.linspace(0, 1, 100)), label="sin(2pix)")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()