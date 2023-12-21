from sklearn.linear_model import LinearRegression
import numpy as np


def get_bootstrap_sample(X, y):
    bootstrap_inds = np.random.choice(len(y), size=len(y), replace=True)
    return X[bootstrap_inds], y[bootstrap_inds]

def get_bs_theta_linreg(X, y, N):
    theta_bs = []

    for _ in range(N):
        X_boot, y_boot = get_bootstrap_sample(X, y)

        model = LinearRegression()
        model.fit(X_boot, y_boot)

        theta_bs.append([model.intercept_] + list(model.coef_))

    return np.array(theta_bs)


class BootstrapLinreg():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def sample(self, k=1):
        return get_bs_theta_linreg(self.X, self.y, k)