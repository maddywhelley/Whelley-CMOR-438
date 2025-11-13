# Perceptron – Linear Classification of Sector Performance

This notebook implements the **Perceptron** algorithm to classify daily market behavior between the **Technology (XLK)** and **Energy (XLE)** ETFs.  
Using short-term return, trend, and volatility features, the perceptron learns a linear decision boundary that separates days when Technology outperforms Energy from the opposite.

As a foundational linear classifier, the perceptron provides a clear baseline for understanding separability in financial data.  
The model’s misclassification curve and decision boundary help illustrate convergence behavior and the limitations of linear methods in noisy market environments.