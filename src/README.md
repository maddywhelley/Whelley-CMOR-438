# Source
This directory contains the core **Python implementations** of selected machine learning algorithms used throughout this repository.  
The modules in `src/rice_ml` are written to be **importable, reusable, and testable**, and they serve as the backend logic for the demonstration notebooks.

Unlike the notebooks, which emphasize visualization and experimentation, the code here focuses on **clean implementations of algorithms** and their underlying mathematics.

---

## Contents

### Implemented Modules

- **Gradient Descent (`gradient_descent.py`)**  
  Implements batch gradient descent for linear regression, including:
  - loss computation,
  - parameter updates,
  - convergence behavior.

- **Logistic Regression (`logistic_regression.py`)**  
  Binary classification using the logistic (sigmoid) function, with:
  - cross-entropy loss,
  - gradient-based optimization,
  - prediction and evaluation utilities.

- **Multilayer Perceptron (`mlp.py`)**  
  A feedforward neural network with one or more hidden layers, featuring:
  - forward propagation,
  - backpropagation,
  - nonlinear activation functions.

- **Principal Component Analysis (`pca.py`)**  
  Dimensionality reduction via orthogonal linear transformations, including:
  - covariance matrix construction,
  - eigenvalue decomposition,
  - projection onto principal components.

- **Package Initialization (`__init__.py`)**  
  Enables the `src` directory to be used as a Python package and exposes selected classes and functions for import.
