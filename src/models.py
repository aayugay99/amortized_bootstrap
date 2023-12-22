class LinearRegressionModel():
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, X):
        return X @ self.theta[1:] + self.theta[0]
 

class LogisticRegressionModel(): 
    def __init__(self, theta):
        self.theta = theta # (n_features + 1) x n_classes

    def __call__(self, X):
        return X @ self.theta[1:, :] + self.theta[0, :]
