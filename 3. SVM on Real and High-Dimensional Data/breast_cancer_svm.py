from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data import load_breast_cancer_data
from preprocessing import split_and_scale


def main():
    X, y, meta = load_breast_cancer_data()

    print("Dataset loaded successfully!")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target classes: {meta.target_names}")

    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    # Linear SVM
    linear_svm = SVC(kernel='linear', C=1.0)
    linear_svm.fit(X_train, y_train)

    y_pred_linear = linear_svm.predict(X_test)
    print("\nLinear SVM Results")
    print("Accuracy:", accuracy_score(y_test, y_pred_linear))
    print("Support Vectors:", linear_svm.n_support_)

    # Polynomial Kernel SVM
    poly_svm = SVC(kernel='poly', degree=2, C=1.0)
    poly_svm.fit(X_train, y_train)

    y_pred_poly = poly_svm.predict(X_test)
    print("\nPolynomial SVM Results")
    print("Accuracy:", accuracy_score(y_test, y_pred_poly))
    print("Support Vectors:", poly_svm.n_support_)


if __name__ == "__main__":
    main()
