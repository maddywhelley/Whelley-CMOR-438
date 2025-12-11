import numpy as np
import pytest
from rice_ml.pca import PCA


# Helper: synthetic dataset generator
def generate_synthetic_data(n_samples=200, random_state=0):
    """
    Generates a simple 3-feature dataset with controlled variance magnitudes.
    Feature 1 has the largest variance, so PCA should identify it first.
    """
    rng = np.random.default_rng(random_state)
    
    x1 = rng.normal(0, 5, n_samples)   # largest variance
    x2 = rng.normal(0, 1, n_samples)
    x3 = rng.normal(0, 0.2, n_samples)

    X = np.vstack([x1, x2, x3]).T
    return X


# TEST 1: PCA fits without runtime errors
def test_pca_fit_runs():
    X = generate_synthetic_data()
    pca = PCA(n_components=2)
    pca.fit(X)
    
    assert pca.components_ is not None
    assert pca.explained_variance_ is not None
    assert pca.mean_ is not None


# TEST 2: Transformed data has correct shape
def test_transform_shape():
    X = generate_synthetic_data()
    pca = PCA(n_components=2)
    
    X_proj = pca.fit_transform(X)
    assert X_proj.shape == (X.shape[0], 2)


# TEST 3: Eigenvalues are returned in descending order
def test_variance_sorting():
    X = generate_synthetic_data()
    pca = PCA(n_components=3)
    pca.fit(X)
    
    ev = pca.explained_variance_
    assert np.all(np.diff(ev) <= 0)  # monotone decreasing


# TEST 4: PCA centers the data correctly
def test_data_centering():
    X = generate_synthetic_data()
    pca = PCA()
    pca.fit(X)
    
    centered = X - pca.mean_
    col_means = centered.mean(axis=0)
    
    # Centered columns should have mean ~ 0
    assert np.allclose(col_means, np.zeros(X.shape[1]), atol=1e-8)


# TEST 5: Calling transform before fit raises error
def test_transform_before_fit():
    X = generate_synthetic_data()
    pca = PCA()
    
    with pytest.raises(ValueError):
        pca.transform(X)


# TEST 6: Using all components preserves dimensionality
def test_full_component_reconstruction_shape():
    X = generate_synthetic_data()
    pca = PCA(n_components=None)
    
    X_proj = pca.fit_transform(X)
    assert X_proj.shape[1] == X.shape[1]


# TEST 7: First principal component corresponds to largest variance direction
def test_first_pc_aligns_with_largest_variance():
    X = generate_synthetic_data()
    pca = PCA(n_components=3)
    pca.fit(X)
    
    # Feature variances
    variances = X.var(axis=0)
    max_variance_feature = np.argmax(variances)

    # First PC loading is the row 0 of components_
    first_pc = pca.components_[0]

    # The feature with largest absolute weight should match the feature with largest variance
    assert np.argmax(np.abs(first_pc)) == max_variance_feature