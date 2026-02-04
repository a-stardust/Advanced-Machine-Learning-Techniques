import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data import generate_high_dimensional_data
from preprocessing import split_and_scale


def main():
    X, y = generate_high_dimensional_data()

    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    # ===============================
    # Linear SVM
    # ===============================
    start_time = time.time()

    linear_svm = SVC(kernel='linear', C=1.0)
    linear_svm.fit(X_train, y_train)

    linear_time = time.time() - start_time
    y_pred_linear = linear_svm.predict(X_test)

    print("Linear SVM Results")
    print("Accuracy:", accuracy_score(y_test, y_pred_linear))
    print("Support Vectors:", linear_svm.n_support_)
    print("Training Time (s):", linear_time)

    # ===============================
    # Polynomial SVM
    # ===============================
    start_time = time.time()

    poly_svm = SVC(kernel='poly', degree=2, C=1.0)
    poly_svm.fit(X_train, y_train)

    poly_time = time.time() - start_time
    y_pred_poly = poly_svm.predict(X_test)

    print("\nPolynomial SVM Results")
    print("Accuracy:", accuracy_score(y_test, y_pred_poly))
    print("Support Vectors:", poly_svm.n_support_)
    print("Training Time (s):", poly_time)


if __name__ == "__main__":
    main()
