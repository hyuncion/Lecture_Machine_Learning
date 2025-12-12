import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

def Problem4():
    # Training data 생성
    np.random.seed(1)
    n_samples = 10
    X = np.sort(np.random.rand(n_samples))
    y = np.sin(2 * np.pi * X) + np.random.randn(n_samples) * 0.1

    # Outlier 생성
    outliers_x = np.array([0.1, 0.3, 0.7])
    outliers_y = np.array([0.5, -0.7, 0.4])
    X = np.concatenate([X, outliers_x])
    y = np.concatenate([y, outliers_y])

    # Alpha, degree 생성
    alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    degrees = [9, 15]

    plt.figure(figsize=(15, 10))

    # Ridge Regression
    for i, degree in enumerate(degrees):
        ax = plt.subplot(2, 2, i+1)
        plt.scatter(X, y)
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = polynomial_features.fit_transform(X.reshape(-1, 1))
        for alpha in alphas:
            modelr = Ridge(alpha=alpha)
            modelr.fit(X_poly, y)
            X_test_poly = polynomial_features.transform(np.linspace(0, 1, 100).reshape(-1, 1))
            y_pred = modelr.predict(X_test_poly)
            plt.plot(np.linspace(0, 1, 100), y_pred, label=f"Ridge, alpha={alpha}")
        plt.legend()
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.title(f"Degree {degree}")

    # Lasso Regression
    for i, degree in enumerate(degrees):
        ax = plt.subplot(2, 2, i+3)
        plt.scatter(X, y)
        for alpha in alphas:
            modell = Lasso(alpha=alpha)
            modell.fit(X_poly, y)
            X_test_poly = polynomial_features.transform(np.linspace(0, 1, 100).reshape(-1, 1))
            y_pred = modell.predict(X_test_poly)
            plt.plot(np.linspace(0, 1, 100), y_pred, label=f"Lasso, alpha={alpha}")
        plt.legend()
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.title(f"Degree {degree}")

    plt.show()
