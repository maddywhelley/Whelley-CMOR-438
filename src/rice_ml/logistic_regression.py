import numpy as np

class LogisticRegression:
    """
    Logistic regression classifier using batch gradient descent.
    
    Parameters
    ----------
    alpha : float
        Learning rate.
    n_iter : int
        Number of gradient descent iterations.
    fit_intercept : bool
        Whether to include an intercept term.
    store_cost : bool
        Whether to store cost history during training.
    
    Attributes
    ----------
    theta_ : ndarray, shape (n_features,)
        Learned parameters.
    cost_history_ : list
        Cost value at each iteration (if store_cost=True).
    """
    def __init__(self, alpha=0.1, n_iter=1000, fit_intercept=True, store_cost=True):
        self.alpha = alpha
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.store_cost = store_cost
        self.theta_ = None
        self.cost_history_ = []
    
    def _add_intercept(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, h, y):
        m = len(y)
        eps = 1e-10  
        return -(1/m) * np.sum(y*np.log(h + eps) + (1 - y)*np.log(1 - h + eps))
    def fit(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        m, n = X.shape
        self.theta_ = np.zeros(n)
        for _ in range(self.n_iter):
            z = X.dot(self.theta_)
            h = self._sigmoid(z)
            
            gradient = (1/m) * X.T.dot(h - y)
            self.theta_ -= self.alpha * gradient
            if self.store_cost:
                cost = self._compute_cost(h, y)
                self.cost_history_.append(cost)
        return self
    
    def predict_proba(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)
        return self._sigmoid(X.dot(self.theta_))
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)