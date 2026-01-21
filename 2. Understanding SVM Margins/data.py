import numpy as np
from sklearn.datasets import make_blobs


def generate_separable_data(n=50, random_state=42):
    X, y = make_blobs(
        n_samples=n,
        centers=2,
        cluster_std=1.0,
        random_state=random_state
    )
    y = np.where(y == 0, -1, 1)
    return X, y


def generate_overlapping_data(n=100, random_state=42):
    X, y = make_blobs(
        n_samples=n,
        centers=2,
        cluster_std=2.5,
        random_state=random_state
    )
    y = np.where(y == 0, -1, 1)
    return X, y
