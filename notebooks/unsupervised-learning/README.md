# Unsupervised Learning
This section of the repository focuses on unsupervised learning methods, where no labeled target variables are 
provided. Instead, the goal is to uncover structure, patterns, or lower-dimensional representations directly from
the data.

The two primary techniques used here are:
- **[K-Means Clustering](./KMeans_Clustering.ipynb)**: a distance-based clustering algorithm
- **[Principal Component Analysis (PCA)](./PCA.ipynb)**: a dimensionality reduction technique based on variance maximization

## K-Means Clustering
### Objective
The goal of K-Means clustering is to partition a dataset into $K$ distinct clusters such that points within a cluster are as close as possible to one another in feature space, while clusters themselves are well separated.

### Method
K-Means operates by iteratively:
1. Initializing $K$ cluster centroids
2. Assigning each data point to the nearest centroid (using Euclidean distance)
3. Updating centroids as the mean of assigned points
4. Repeating until convergence

The algorithm minimizes the within-cluster sum of squared distances (inertia).

We use the wine quality dataset from `sklearn.datasets`.

## Principle Component Analysis (PCA) 
### Objective
The goal of PCA is to reduce the dimensionality of a dataset while preserving as much variance as possible. This is particularly useful for visualization, noise reduction, and preprocessing before other learning tasks.

### Method
PCA works by:
1. Centering the data
2. Computing the covariance matrix
3. Finding its eigenvalues and eigenvectors
4. Projecting the data onto the directions of maximum variance (principal components)

Each principal component is orthogonal to the others and ordered by explained variance.
