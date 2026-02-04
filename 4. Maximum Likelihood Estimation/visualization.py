import matplotlib.pyplot as plt


def plot_likelihood(x_vals, y_vals, xlabel, ylabel, title):
    plt.figure(figsize=(7, 5))
    plt.plot(x_vals, y_vals)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()
