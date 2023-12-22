from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np


def get_bootstrap_sample(X, y, stratify=False):
    inds = np.arange(len(y))
    bootstrap_inds = []
    if stratify:
        for label in np.unique(y):
            bootstrap_inds += list(np.random.choice(inds[y == label], size=int((y == label).sum()), replace=True))
    else:
        bootstrap_inds = np.random.choice(range(len(y)), size=len(y), replace=True)
    return X[bootstrap_inds], y[bootstrap_inds]

def get_bs_theta_linreg(X, y, N):
    theta_bs = []

    for _ in range(N):
        X_boot, y_boot = get_bootstrap_sample(X, y)

        model = LinearRegression()
        model.fit(X_boot, y_boot)

        theta_bs.append([model.intercept_] + list(model.coef_))

    return np.array(theta_bs)

def get_bs_theta_logreg(X, y, N):
    theta_bs = []

    for _ in range(N):
        X_boot, y_boot = get_bootstrap_sample(X, y, True)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_boot, y_boot)

        theta_bs.append(np.hstack([model.intercept_.reshape(-1, 1), model.coef_]).T)
    
    return np.stack(theta_bs, axis=0)

class BootstrapLinreg():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def sample(self, k=1):
        return get_bs_theta_linreg(self.X, self.y, k)
    
class BootstrapLogreg():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def sample(self, k=1):
        return get_bs_theta_logreg(self.X, self.y, k)