# Multilayer Perceptron (MLP)

This notebook demonstrates the **Multilayer Perceptron**, a neural network capable of modeling nonlinear relationships and decision boundaries.  
We apply an MLP to the two-moons dataset to highlight why neural networks outperform linear classifiers on problems that are not linearly separable.

The model is implemented **from scratch**, including forward propagation, backpropagation, and gradient-descent training.  
With one hidden layer, the MLP learns a flexible decision surface that adapts to the curved structure of the data, achieving high accuracy and a stable, steadily decreasing loss curve.

The final decision boundary and residual classification errors illustrate the MLP’s ability to generalize nonlinear patterns that a single perceptron cannot capture.