# Tests

This directory contains **unit tests** for the core machine learning implementations in the `src/` directory.  
All tests are written using **`pytest`** and are designed to verify correctness, numerical behavior, and expected model outputs.

The test suite is automatically executed on every **pull request** and **push to the main branch** via continuous integration (CI), ensuring code reliability and preventing regressions.

---

## Tested Modules

The following implementations are covered by unit tests:

- **Gradient Descent**
  - parameter updates
  - convergence on simple datasets
  - loss reduction over iterations

- **Logistic Regression**
  - correct probability outputs
  - classification accuracy on linearly separable data
  - stable training behavior

- **Multilayer Perceptron (MLP)**
  - forward pass correctness
  - training loss decrease
  - prediction shape and validity

- **Principal Component Analysis (PCA)**
  - correct dimensionality reduction
  - variance ordering of components
  - orthogonality of principal axes
