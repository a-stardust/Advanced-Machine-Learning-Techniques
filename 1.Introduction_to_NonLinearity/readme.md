# Linear and Non-Linear Classification Techniques

This repository explores fundamental classification challenges in machine learning, focusing on the transition from linear to non-linear decision boundaries using **Feature Engineering** and **Kernel Methods**.

---

### 📂 Project Structure

* **`data.py`**: Synthetic data generation for linearly separable and XOR datasets.
* **`visual.py`**: Visualization utilities for 2D data plotting and decision boundary analysis.
* **`task0.py`**: Implementation of classification on linearly separable data.
* **`task1.py`**: Analysis of non-linearly separable (XOR) datasets.
* **`task2.py`**: Solving non-linear problems via **Polynomial Feature Transformation** with Logistic Regression.
* **`task3.py`**: Comparative analysis of **Support Vector Machine (SVM)** kernels, including Linear, Polynomial, and RBF.

---

### 🚀 Key Technical Implementation

#### Feature Transformation
A standard linear model (Logistic Regression) typically achieves ~50% accuracy on XOR data due to the lack of linear separability. By applying `PolynomialFeatures(degree=2)`, the input space is transformed, allowing the linear model to achieve ~100% accuracy.



#### SVM Kernel Analysis
This project demonstrates the "Kernel Trick" using Scikit-Learn's `SVC`:
* **Linear Kernel**: Unable to resolve the XOR class distribution.
* **Polynomial Kernel**: Successfully captures non-linear relationships with $degree=2$.
* **RBF Kernel**: Utilizes Gaussian radial basis functions to effectively map and classify complex boundaries.



---

### 🛠️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-username/repository-name.git](https://github.com/your-username/repository-name.git)
