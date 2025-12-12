import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def Problem5():
    # Generate 100 samples
    np.random.seed(1)
    n_samples = 100
    X = np.sort(np.random.rand(n_samples))
    y = np.sin(2 * np.pi * X)

    # Generate regression lines with polynomial basis functions
    degrees = [1, 3, 5, 9, 15]
    plt.figure(figsize=(10, 5))
    for degree in degrees:
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = polynomial_features.fit_transform(X.reshape(-1, 1))
        model = LinearRegression()
        model.fit(X_poly, y)
        X_test = np.linspace(0, 1, 100).reshape(-1, 1)
        X_test_poly = polynomial_features.transform(X_test)
        y_pred = model.predict(X_test_poly)
        plt.plot(X_test, y_pred, label="degree %d" % degree)

    # Plot original sin(2πx) function
    plt.plot(X, y, 'o', label="samples")
    plt.plot(np.linspace(0, 1, 100), np.sin(2 * np.pi * np.linspace(0, 1, 100)), label="sin(2pix)")
    plt.xlim(0, 1)
    plt.ylim(-2, 2)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()