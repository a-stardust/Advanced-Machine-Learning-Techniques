import numpy as np

def generate_gmm_data():
    """Generate two toy clusters for GMM from scratch."""
    np.random.seed(42)

    # Cluster 1
    mean1 = [4.5, 1.8]
    cov1 = [[0.2, 0.05],
            [0.05, 0.1]]
    data1 = np.random.multivariate_normal(mean1, cov1, 100)

    # Cluster 2
    mean2 = [6.0, 2.5]
    cov2 = [[0.3, -0.04],
            [-0.04, 0.2]]
    data2 = np.random.multivariate_normal(mean2, cov2, 100)

    # Combine
    X = np.vstack((data1, data2))
    return X
