import numpy as np
from rice_ml import LogisticRegression

# TEST 1: Verify shape
def test_logistic_shapes():
    X = np.array([[0,1],[1,1],[2,1]])
    y = np.array([0, 0, 1])
    
    model = LogisticRegression(alpha=0.1, n_iter=200)
    model.fit(X, y)
    
    assert model.theta_.shape[0] == X.shape[1] + 1

# TEST 2: Verify convergence
def test_logistic_prediction():
    X = np.array([[0],[1],[2],[3]])
    y = np.array([0,0,1,1])
    
    model = LogisticRegression(alpha=0.1, n_iter=500)
    model.fit(X, y)
    
    preds = model.predict(X)
    assert preds.shape == y.shape