import numpy as np


def generate_gaussian_data(mu=2.0, sigma=1.5, n=1000, random_state=42):
    np.random.seed(random_state)
    return np.random.normal(mu, sigma, n)


def generate_bernoulli_data(p=0.7, n=500, random_state=42):
    np.random.seed(random_state)
    return np.random.binomial(1, p, n)
