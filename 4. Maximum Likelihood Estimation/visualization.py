import matplotlib.pyplot as plt


def plot_likelihood(x_vals, y_vals, xlabel, ylabel, title):
    plt.figure(figsize=(7, 5))
    plt.plot(x_vals, y_vals)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_convergence(sample_sizes, mle_values, true_value, xlabel, ylabel, title):
    plt.figure(figsize=(7, 5))
    plt.plot(sample_sizes, mle_values, marker='o')
    plt.axhline(y=true_value, color='r', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()
