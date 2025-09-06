import numpy as np

class ElasticNetRegressor:
    def __init__(self, learning_rate=0.01, n_iterations=1000, alpha=0.5, l1_ratio=0.5, tolerance=1e-6, early_stop=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha             
        self.l1_ratio = l1_ratio      
        self.tolerance = tolerance
        self.early_stop = early_stop
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _elastic_net_loss(self, y_true, y_pred, weights):
        mse = np.mean((y_true - y_pred) ** 2)
        l1 = np.sum(np.abs(weights))
        l2 = np.sum(weights ** 2)
        return mse + self.alpha * (self.l1_ratio * l1 + (1 - self.l1_ratio) * l2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y

            # اشتقاق MSE
            dw_mse = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            # اشتقاق L1 (sign)
            dw_l1 = self.alpha * self.l1_ratio * np.sign(self.weights)

            # اشتقاق L2
            dw_l2 = self.alpha * (1 - self.l1_ratio) * 2 * self.weights

            # المجموع النهائي
            dw = dw_mse + dw_l1 + dw_l2

            # التحديث
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # حساب الخسارة
            cost = self._elastic_net_loss(y, y_pred, self.weights)
            self.cost_history.append(cost)

            if self.early_stop and i > 0:
                if abs(self.cost_history[-2] - cost) < self.tolerance:
                    print(f"Stopped early at iteration {i}")
                    break

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def get_cost_history(self):
        return self.cost_history



if __name__ == "__main__":
    
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

   
    model = ElasticNetRegressor(
        learning_rate=0.05,
        n_iterations=1000,
        alpha=0.1,
        l1_ratio=0.7
    )

    
    model.fit(X, y)

    
    print("Weights:", model.weights)
    print("Bias:", model.bias)
    print("First 10 Cost Values:", model.get_cost_history()[:10])

    
    X_test = np.array([[6], [7]])
    predictions = model.predict(X_test)
    print("Predictions:", predictions)