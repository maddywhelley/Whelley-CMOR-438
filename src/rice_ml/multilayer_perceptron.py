import numpy as np

# Activation functions

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# Multilayer perceptron
class MLP:
    """
    A simple multilayer perceptron (MLP).
    
    Input layer: 2 features
    Hidden layer: hidden_dim neurons (ReLU)
    Output layer: 1 neuron (sigmoid)
    
    Training: 
        Loss: Binary Cross-Entropy
        Optimization: Gradient descent
    """
    def __init__(self, hidden_dim=8, lr=0.01, epochs=5000, verbose=True):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
    
    # Weight initialization
    def initialize_weights(self, input_dim):
        self.W1 = np.random.randn(input_dim, self.hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        
        self.W2 = np.random.randn(self.hidden_dim, 1) * np.sqrt(2 / self.hidden_dim)
        self.b2 = np.zeros((1, 1))
    
    # Forward pass
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)
        
        self.z2 = self.h1 @ self.W2 + self.b2
        self.out = sigmoid(self.z2)
        return self.out
    
    # Loss function
    def compute_loss(self, y_hat, y):
        m = len(y)
        y = y.reshape(-1, 1)
        eps = 1e-8
        return -(1/m) * np.sum(
            y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
        )
    
    # Backpropagation
    def backward(self, X, y, y_hat):
        m = len(y)
        y = y.reshape(-1, 1)
        
        dz2 = y_hat - y
        dW2 = (1/m) * self.h1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        dz1 = (dz2 @ self.W2.T) * relu_deriv(self.z1)
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Gradient descent update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
    
    # Training loop
    def fit(self, X, y):
        self.initialize_weights(X.shape[1])
        self.loss_history = []
        
        for epoch in range(self.epochs):
            y_hat = self.forward(X)
            loss = self.compute_loss(y_hat, y)
            self.loss_history.append(loss)
            
            self.backward(X, y, y_hat)
            
            if self.verbose and epoch % 500 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    # Prediction
    def predict(self, X):
        y_hat = self.forward(X)
        return (y_hat > 0.5).astype(int)