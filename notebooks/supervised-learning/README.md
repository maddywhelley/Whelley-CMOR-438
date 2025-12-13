# Supervised Learning
Supervised learning is a core branch of machine learning where a model is trained on labeled data to learn the mapping from inputs to *known* outputs. 
This section of the repository implements and visualizes several supervised learning algorithms including linear models, instance-based learning, neural networks, and tree-based approaches.

---

## Contents
The supervised learning methods covered in this directory include:

### Linear and Gradient-Based Models
- **[Linear Regression](./notebooks/Linear_Regression.ipynb)**: Ordinary least squares regression for continuous targets.
- **Gradient Descent**: Batch gradient descent applied to linear regression, emphasizing optimization dynamics and convergence behavior.
- **[Logistic Regression](./notebooks/Logistic_Regression.ipynb)**: Binary classification using the logistic function and cross-entropy loss.

### Instance-Based Learning
- **[K Nearest Neighbors](./notebooks/KNN.ipynb)**: A non-parametric method for classification and regression based on distance metrics and local neighborhoods.

### Neural Networks
- **[Perceptron](./notebooks/Perceptron.ipynb)**: A single-layer linear classifier used to motivate the limitations of linear decision boundaries.
- **[Multilayer Perceptron](./notebooks/Multilayer_Perceptron.ipynb)**: A feedforward neural network with one or more hidden layers, trained via backpropagation to model nonlinear decision boundaries.

### Support Vector Methods
- **[Support Vector Regression (SVR)](./notebooks/SVM_regression.ipynb)**: Regression using margin maximization and kernel methods to capture nonlinear structure.

### Tree-Based Models
- **[Decision Tree Regression](./notebooks/Dicsion_Trees.ipynb)**: A recursive, rule-based regression model using feature splits to minimize error.
- **[Ensemble Learning](./notebooks/Ensemble_Learning.ipynb)**: Methods that combine multiple weak learners (e.g., bagging or random forests) to improve stability and predictive performance.

## Structure
Each algorithm follows roughly the same structure: 
- Conceptual overview explaining the model and its assumptions
- Implementation (from scratch where feasible, otherwise using `scikit-learn`)
- Training and evaluation using appropriate metrics
- Visualization of predictions, decision boundaries, or error behavior when informative

