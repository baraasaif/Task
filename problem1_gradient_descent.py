import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-6, early_stop=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.early_stop = early_stop
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            error = y_predicted - y

            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            cost = (1 / (2 * n_samples)) * np.sum(error ** 2)
            self.cost_history.append(cost)

            if self.early_stop and i > 0:
                if abs(self.cost_history[-2] - cost) < self.tolerance:
                    print(f"Stopped early at iteration {i}")
                    break

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def get_cost_history(self):
        return self.cost_history


#  تجربة التدريب والتوقع
if __name__ == "__main__":
    
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

  
    model = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

   
    print("Weights:", model.weights)
    print("Bias:", model.bias)
    print("Cost history:", model.get_cost_history()[:10])  # أول 10 قيم فقط

    
    X_test = np.array([[6], [7]])
    predictions = model.predict(X_test)
    print("Predictions:", predictions)