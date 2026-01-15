

from sklearn.svm import SVC
from data import generate_xor_data
from visual import plot_2d_datat
from sklearn.metrics import accuracy_score

def main():
    X,y= generate_xor_data()
    plot_2d_datat(X,y,title="XOR Data (Original Space)")

    #linear SVM
    linear_scm=SVC(kernel='linear')
    linear_scm.fit(X,y)
    y_pred_linear=linear_scm.predict(X)
    plot_2d_datat(X,y_pred_linear,title="XOR Data - Linear SVM Predictions (fails)")

    #polynomial kernel SVM
    poly_svm=SVC(kernel='poly',degree=2)
    poly_svm.fit(X,y)
    y_pred_poly=poly_svm.predict(X)
    plot_2d_datat(X,y_pred_poly,title="XOR Data - Polynomial Kernel SVM Predictions ( Succeeds)")

    #rbf kernel SVM
    rbf_svm=SVC(kernel='rbf', gamma='scale')
    rbf_svm.fit(X,y)
    y_pred_rbf=rbf_svm.predict(X)
    plot_2d_datat(X,y_pred_rbf,title="XOR Data - RBF Kernel SVM Predictions (Succeeds)")

    #print accuracies
    print("Linear SVM Accuracy:", accuracy_score(y,y_pred_linear))
    print("Polynomial Kernel SVM Accuracy:", accuracy_score(y,y_pred_poly))
    print("RBF Kernel SVM Accuracy:", accuracy_score(y,y_pred_rbf))


if __name__ == "__main__":
    main()



##Apply differnet kernels , 2 more push all to github
        