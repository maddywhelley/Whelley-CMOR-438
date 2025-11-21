# Support Vector Regression (SVR) – Predicting California Housing Prices

This notebook applies **Support Vector Regression (SVR)** to the California Housing dataset to model the nonlinear relationships between demographic and geographic features and **median house value**.

SVR provides a margin-based approach to regression, using an ε-insensitive loss to control sensitivity to noise and kernel functions to capture complex structure in the data.  
We compare a **Linear SVR** model with an **RBF-kernel SVR**, highlighting how kernel choice affects predictive performance.

The RBF model shows a significantly stronger ability to follow the true value distribution, producing tighter predictions and more symmetric residuals.  
In contrast, the linear model underfits, reflecting its inability to represent nonlinear interactions among housing features.