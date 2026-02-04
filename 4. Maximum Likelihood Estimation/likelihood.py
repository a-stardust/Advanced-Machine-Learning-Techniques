import numpy as np


def gaussian_log_likelihood(x, mu, sigma):
    n = len(x)
    return -n * np.log(sigma) - (1 / (2 * sigma**2)) * np.sum((x - mu)**2)


def bernoulli_log_likelihood(x, p):
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    return np.sum(x * np.log(p) + (1 - x) * np.log(1 - p))
