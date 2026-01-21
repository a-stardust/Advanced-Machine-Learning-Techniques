import numpy as np
import matplotlib.pyplot as plt


def plot_svm(X, y, w, b, support_vectors=None, title="SVM Plot"):
    plt.figure(figsize=(7, 6))

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")

    x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, 'k-')

    margin = 1 / np.linalg.norm(w)
    y_margin_pos = -(w[0] * x_vals + b - 1) / w[1]
    y_margin_neg = -(w[0] * x_vals + b + 1) / w[1]
    plt.plot(x_vals, y_margin_pos, 'k--')
    plt.plot(x_vals, y_margin_neg, 'k--')

    if support_vectors is not None:
        plt.scatter(
            support_vectors[:, 0],
            support_vectors[:, 1],
            s=120,
            facecolors='none',
            edgecolors='k'
        )

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
