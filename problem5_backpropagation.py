import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='sigmoid', learning_rate=0.1, n_iterations=10000):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation = activation
        self.weights = []
        self.biases = []

        
        if activation == 'sigmoid':
            self.act = sigmoid
            self.act_deriv = sigmoid_derivative
        elif activation == 'relu':
            self.act = relu
            self.act_deriv = relu_derivative
        else:
            raise ValueError("Unsupported activation")

    
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def fit(self, X, y):
        for iteration in range(self.n_iterations):
            # الانتشار الأمامي
            activations = [X]
            for w, b in zip(self.weights, self.biases):
                z = np.dot(activations[-1], w) + b
                a = self.act(z)
                activations.append(a)

            #  حساب الخطأ
            error = activations[-1] - y
            deltas = [error * self.act_deriv(activations[-1])]

            #  الانتشار الخلفي
            for i in reversed(range(len(self.weights) - 1)):
                delta = np.dot(deltas[0], self.weights[i+1].T) * self.act_deriv(activations[i+1])
                deltas.insert(0, delta)

            #  تحديث الأوزان والانحرافات
            for i in range(len(self.weights)):
                dw = np.dot(activations[i].T, deltas[i])
                db = np.sum(deltas[i], axis=0, keepdims=True)
                self.weights[i] -= self.learning_rate * dw
                self.biases[i] -= self.learning_rate * db

            
            if iteration % 1000 == 0:
                loss = np.mean(np.square(error))
                print(f"Iteration {iteration}, Loss: {loss:.4f}")

    def predict(self, X):
        a = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = self.act(z)
        return a


 
if __name__ == "__main__":
    # بيانات XOR
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # إنشاء النموذج: 2 مدخلات → 4 مخفية → 1 مخرج
    nn = NeuralNetwork(layer_sizes=[2, 4, 1], activation='sigmoid', learning_rate=0.5, n_iterations=10000)
    nn.fit(X, y)

    
    predictions = nn.predict(X)
    print("Predictions:")
    print(np.round(predictions, 3))