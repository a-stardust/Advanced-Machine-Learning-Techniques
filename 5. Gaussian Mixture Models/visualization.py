import matplotlib.pyplot as plt

def plot_gmm_scratch(X, responsibilities, means):
    """Plot the clusters learned from scratch GMM."""
    plt.scatter(X[:,0], X[:,1], c=responsibilities[:,0], cmap='coolwarm')
    plt.scatter(means[:,0], means[:,1], c='black', marker='X', s=200)
    plt.title("Cluster Learning using EM")
    plt.show()
