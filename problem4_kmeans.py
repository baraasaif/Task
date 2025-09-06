import numpy as np

class KMeansClustering:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, distance_metric='euclidean', random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.distance_metric = distance_metric
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def _distance(self, X, centroids):
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        else:
            raise ValueError("Unsupported distance metric")

    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        centroids = [X[np.random.randint(n_samples)]]

        for _ in range(1, self.n_clusters):
            dist_sq = np.min(self._distance(X, np.array(centroids)) ** 2, axis=1)
            probs = dist_sq / dist_sq.sum()
            next_idx = np.random.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)

        for i in range(self.max_iter):
            distances = self._distance(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else self.centroids[j]
                for j in range(self.n_clusters)
            ])

            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids

            if shift < self.tol:
                print(f"Converged at iteration {i}")
                break

        self.labels_ = labels

    def predict(self, X):
        distances = self._distance(X, self.centroids)
        return np.argmin(distances, axis=1)

    def get_centroids(self):
        return self.centroids

    def get_labels(self):
        return self.labels_



if __name__ == "__main__":
    
    X = np.array([
        [1, 2], [1.5, 1.8], [2, 2.2],
        [8, 8], [8.5, 7.5], [9, 8.2],
        [0, 9], [0.5, 8.5], [1, 9.5]
    ])

    
    model = KMeansClustering(n_clusters=3, random_state=42)
    model.fit(X)

    
    print("Centroids:")
    print(model.get_centroids())

    print("Labels:")
    print(model.get_labels())

    
    X_test = np.array([[2, 2], [9, 9], [0, 10]])
    predictions = model.predict(X_test)
    print("Predictions for new points:", predictions)