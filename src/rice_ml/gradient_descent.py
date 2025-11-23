import numpy as np
from typing import Optional

class GradientDescentRegressor:
    """ 
    A simple linear regression model trained using batch gradient descent.
    
    Parameters:
    alpha (float) : 
        Learning rate / step size
    n_iter (int) : 
        Number of gradient descent iterations
    fit_intercept (bool) : 
        Whether to include a bias term
    store_cost (bool) : 
        Whether to store the MSE cost at each iteration
    """
    def __init__(self, alpha: float = 0.01, n_iter: int = 1000, fit_intercept: bool = True,
                store_cost: bool = False):
        self.alpha = alpha
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.store_cost = store_cost

        # Initialize attributes
        self.theta_ = None
        self.cost_history_ = None
    
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        return np.c_[np.ones((X.shape[0], 1)), X] if self.fit_intercept else X
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the linear model using batch gradient descent.

        Parameters
        
        X (np.ndarray) : 
            Feature matrix of shape (n_samples, n_features).
        y (np.ndarray) : 
            Target vector of shape (n_samples,).
        
        Returns
        self : object
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        X_b = self._add_bias(X)
        m, n = X_b.shape
        
        # Initialize weights
        self.theta_ = np.zeros(n)
        if self.store_cost:
            self.cost_history_ = []
            
        # Gradient descent loop
        for _ in range(self.n_iter):
            y_pred = X_b.dot(self.theta_)
            error = y_pred - y
            
            # Compute gradient
            gradient = (2 / m) * X_b.T.dot(error)
            
            # Update step
            self.theta_ -= self.alpha * gradient
            
            # Store cost if enabled
            if self.store_cost:
                mse = np.mean(error ** 2)
                self.cost_history_.append(mse)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values using the learned model.
        
        Parameters
        X (np.ndarray) : 
            Feature matrix of shape (n_samples, n_features).
        
        Returns
        np.ndarray : 
            Predicted values of shape (n_samples,).
        """
        if self.theta_ is None:
            raise ValueError("Model has not been fitted yet.")
        X = np.array(X, dtype=float)
        X_b = self._add_bias(X)
        return X_b.dot(self.theta_)