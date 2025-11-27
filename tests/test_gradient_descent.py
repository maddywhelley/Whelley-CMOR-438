import numpy as np
import pytest
from rice_ml import GradientDescentRegressor

# Helper: synthetic linear dataset generator
def generate_synthetic_data(n_samples=300, noise=0.1, random_state=42):
    """
    Creates a synthetic linear dataset:
        y = 3*x1 - 2*x2 + intercept + noise
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, 2))
    
    true_w = np.array([3.0, -2.0]) # true weights
    intercept = 4.5 # true intercept
    
    y = X @ true_w + intercept + noise * rng.normal(size=n_samples)
    return X, y, true_w, intercept

# TEST 1: Model fits without runtime errors
def test_fit_runs(): 
    X, y, _, _ = generate_synthetic_data()
    model = GradientDescentRegressor(alpha=0.05, n_iter=200)
    model.fit(X, y)
    assert model.theta_ is not None

# TEST 2: Predictions have correct shape
def test_prediction_shape():
    X, y, _, _ = generate_synthetic_data()
    model = GradientDescentRegressor(alpha=0.05, n_iter=200)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

# TEST 3: Cost should decrease over iterations
def test_cost_decreases():
    X, y, _, _ = generate_synthetic_data()
    model = GradientDescentRegressor(alpha=0.05, n_iter=250, store_cost=True)
    
    model.fit(X, y)
    cost_history = model.cost_history_
    
    assert len(cost_history) == 250
    # First cost > last cost
    assert cost_history[0] > cost_history[-1]

# TEST 4: Learned weights close to true parameters
def test_parameter_convergence():
    X, y, true_w, intercept = generate_synthetic_data(noise=0.0)
    
    model = GradientDescentRegressor(alpha=0.1, n_iter=500)
    model.fit(X, y)
    
    # Extract learned parameters
    learned_intercept = model.theta_[0]
    learned_weights = model.theta_[1:]
    
    # Expect close convergence
    assert np.allclose(learned_weights, true_w, atol=0.2)
    assert np.isclose(learned_intercept, intercept, atol=0.2)

# TEST 5: Model raises error if predicting before fitting
def test_predict_before_fit():
    X, y, _, _ = generate_synthetic_data()
    model = GradientDescentRegressor()
    
    with pytest.raises(ValueError):
        model.predict(X)

# TEST 6: fit_intercept flag works, i.e. no intercept term
def test_no_intercept_behavior():
    X, y, _, _ = generate_synthetic_data()
    model = GradientDescentRegressor(fit_intercept=False, alpha=0.05, n_iter=200)
    
    model.fit(X, y)
    
    # theta_ should have length equal to number of features, not +1
    assert model.theta_.shape[0] == X.shape[1]