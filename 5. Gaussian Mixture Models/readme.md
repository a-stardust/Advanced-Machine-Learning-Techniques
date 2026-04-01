# 5. Gaussian Mixture Models

This module demonstrates **Gaussian Mixture Models (GMM)** and the Expectation-Maximization (EM) algorithm.

## Files
- `data.py`: Generates toy data for GMM clustering.
- `gmm_scratch.py`: Implements the EM algorithm for GMM from scratch.
- `visualization.py`: Contains plotting functions for the clusters.
- `main_scratch.py`: Runs the toy EM algorithm implementation.
- `gmm_clustering.py`: Applies scikit-learn's `GaussianMixture` on a real marketing campaign dataset.
- `GMM.ipynb`, `GMM (1).ipynb`, `Univariate_Gaussian.ipynb`, `gmm_on_dataset.ipynb`: Original lab notebooks containing the experiments and explanations.

## How to Run

To run the from-scratch EM implementation:
```bash
python main_scratch.py
```

To run the scikit-learn implementation on the marketing dataset:
```bash
python gmm_clustering.py
```
