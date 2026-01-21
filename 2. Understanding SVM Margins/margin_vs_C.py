import matplotlib.pyplot as plt
from sklearn.svm import SVC
from data import generate_overlapping_data


def main():
    X, y = generate_overlapping_data()
    C_values = [0.01, 0.1, 1, 10, 100]

    fig, axes = plt.subplots(1, len(C_values), figsize=(18, 5))

    for ax, C in zip(axes, C_values):
        svm = SVC(kernel='linear', C=C)
        svm.fit(X, y)

        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
        ax.set_title(f"C = {C}")

    plt.suptitle("Effect of Regularization Parameter C on SVM Margin")
    plt.show()


if __name__ == "__main__":
    main()
