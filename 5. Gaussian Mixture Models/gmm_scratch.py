import numpy as np
from scipy.stats import multivariate_normal

def fit_gmm_em(X, k=2, iterations=2):
    """Fit GMM using EM algorithm from scratch."""
    n, d = X.shape

    # Initial means
    means = X[np.random.choice(n, k, replace=False)]

    # Initial covariance matrices
    covariances = [np.cov(X.T) for _ in range(k)]

    # Initial mixing coefficients
    weights = np.ones(k) / k

    for step in range(iterations):
        # E-step
        responsibilities = np.zeros((n, k))
        for i in range(k):
            rv = multivariate_normal(means[i], covariances[i])
            responsibilities[:, i] = weights[i] * rv.pdf(X)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # M-step
        Nk = responsibilities.sum(axis=0)
        weights = Nk / n
        means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]

        covariances = []
        for i in range(k):
            diff = X - means[i]
            cov = np.dot(responsibilities[:, i] * diff.T, diff) / Nk[i]
            covariances.append(cov)

    return responsibilities, means, covariances, weights
