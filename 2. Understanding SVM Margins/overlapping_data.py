from sklearn.svm import SVC
from data import generate_overlapping_data
from visual import plot_svm


def main():
    X, y = generate_overlapping_data()

    svm = SVC(kernel='linear', C=1e6)
    svm.fit(X, y)

    w = svm.coef_[0]
    b = svm.intercept_[0]

    print("Number of support vectors (hard margin):", len(svm.support_vectors_))

    plot_svm(
        X, y, w, b,
        support_vectors=svm.support_vectors_,
        title="Hard Margin SVM on Overlapping Data"
    )


if __name__ == "__main__":
    main()
