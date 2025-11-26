import numpy as np
from rice_ml.multilayer_perceptron import MLP

# TEST 1: Initialization produces correctly shaped weights
def test_mlp_initialization():
    model = MLP(hidden_dim=8, lr=0.01, epochs=10, verbose=False)
    X_dummy = np.random.randn(5, 2) # 2 input features
    y_dummy = np.random.randint(0, 2, size=5)
    
    model.fit(X_dummy, y_dummy)
    
    # Check shapes
    assert model.W1.shape == (2, 8)
    assert model.b1.shape == (1, 8)
    assert model.W2.shape == (8, 1)
    assert model.b2.shape == (1, 1)


# TEST 2: Forward pass returns probabilities in [0, 1]
def test_mlp_forward_pass_output_range():
    model = MLP(hidden_dim=4, lr=0.1, epochs=5, verbose=False)
    X = np.random.randn(10, 2)
    y = np.random.randint(0, 2, size=10)
    
    model.fit(X, y)
    out = model.forward(X)
    
    assert out.shape == (10, 1)
    assert np.all(out >= 0) and np.all(out <= 1)


# TEST 3: Loss decreases over training
def test_mlp_loss_decreases():
    np.random.seed(0)
    model = MLP(hidden_dim=4, lr=0.05, epochs=200, verbose=False)
    
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # linearly related dummy target
    
    model.fit(X, y)
    
    assert model.loss_history[0] > model.loss_history[-1]


# TEST 4: predict() outputs binary labels of correct shape
def test_mlp_predict_output():
    model = MLP(hidden_dim=6, lr=0.01, epochs=20, verbose=False)
    X = np.random.randn(20, 2)
    y = np.random.randint(0, 2, size=20)
    
    model.fit(X, y)
    preds = model.predict(X)
    
    assert preds.shape == (20, 1)
    assert set(np.unique(preds)).issubset({0, 1})


# TEST 5: Gradients actually update weights
def test_mlp_weights_update():
    model = MLP(hidden_dim=4, lr=0.1, epochs=1, verbose=False)
    X = np.random.randn(5, 2)
    y = np.random.randint(0, 2, size=5)
    
    model.initialize_weights(input_dim=2)
    
    W1_before = model.W1.copy()
    W2_before = model.W2.copy()
    
    # One training step
    y_hat = model.forward(X)
    model.backward(X, y, y_hat)
    
    # After backward, weights must change
    assert not np.allclose(W1_before, model.W1)
    assert not np.allclose(W2_before, model.W2)


# TEST 6: MLP can fit a trivial dataset
def test_mlp_can_fit_tiny_dataset():
    model = MLP(hidden_dim=3, lr=0.1, epochs=2000, verbose=False)
    
    # XOR-like small dataset (nonlinear)
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0, 1, 1, 0])
    
    model.fit(X, y)
    
    preds = model.predict(X).flatten()
    # Because of nonlinearity, model should get at least 3/4 correct
    assert (preds == y).sum() >= 3