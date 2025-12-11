import numpy as np

class PCA:
    """Principle Component Analysis (PCA) implemented from scratch using NumPy.
    This class computes the principal axes of variation and provides a method
    to project data into a lower-dimensional subspace.
    
    Attributes: 
    components_: np.ndarray
        Matrix whose rows are principle axes (eigenvectors).
    explained_variance_: np.ndarray
        Eigenvalues corresponding to the principle components 
    mean_: np.ndarray
        Mean of the data used for centering.
    """
    
    def __init__(self, n_components=None):
        """
        Parameters:
            n_components (int, optional): Number of principle components to
            retain. Defaults to None. If None, all components are kept.
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
    
    def fit(self, X):
        """Fit PCA on dataset X.
        
        Steps:
        1.Center data.
        2. Compute covariance matrix.
        3. Eigen-decompose covariance.
        4. Sort eigenvalues (descending order).
        5. Store principal components.

        Args:
            X (np.ndarray): Shape (n_samples, n_features).
        Returns:
            self
        """
        # Step 1: Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Step 2: Covariance matrix
        covariance = np.cov(X_centered, rowvar=False)
        
        # Step 3: Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # Step 4: Sort eigenvalues/eigenvectors descending
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # Step 5: Keep n_components if specified
        if self.n_components is not None:
            eigenvalues = eigenvalues[: self.n_components]
            eigenvectors = eigenvectors[:, : self.n_components]
        self.explained_variance_ = eigenvalues
        self.components_ = eigenvectors.T
        
        return self
    
    def transform(self, X):
        """Project the data X onto the learned principle components.
        
        Args:
            X (np.ndarray)
        Returns:
            X_proj (np.ndarray)
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before calling transform().")
        
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X):
        """
        Convenience function: fit PCA and return projected data.
        """
        self.fit(X)
        return self.transform(X)
