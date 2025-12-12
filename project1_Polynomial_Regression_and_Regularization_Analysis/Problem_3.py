import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def Problem3():
    # 데이터 생성
    np.random.seed(1)
    n_samples = 10
    X = np.sort(np.random.rand(n_samples))
    y = np.sin(2 * np.pi * X) + np.random.randn(n_samples) * 0.1

    # outlier 생성
    outliers_x = np.array([0.1, 0.3, 0.7])
    outliers_y = np.array([0.5, -0.7, 0.4])
    X = np.concatenate([X, outliers_x])
    y = np.concatenate([y, outliers_y])

    degrees = [1, 3, 5, 9, 15]

    # 그래프 그리기
    plt.figure(figsize = (15, 10))

    for i, degree in enumerate(degrees):
        ax = plt.subplot(2, 3, i + 1)
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = polynomial_features.fit_transform(X.reshape(-1, 1))
        model = LinearRegression()
        model.fit(X_poly, y)
        X_test = np.linspace(0, 1, 100).reshape(-1, 1)
        X_test_poly = polynomial_features.transform(X_test)
        y_pred = model.predict(X_test_poly)
        plt.scatter(X, y)
        plt.plot(X_test, y_pred, label="degree %d" % degree)
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.legend()
        plt.title(f"degree {degree}")
        plt.xlabel("X")
        plt.ylabel("y")

    plt.show()