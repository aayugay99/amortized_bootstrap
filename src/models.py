class LinearModel():
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, X):
        return X @ self.theta[1:] + self.theta[0]
 