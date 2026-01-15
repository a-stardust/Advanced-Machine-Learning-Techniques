import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from data import generate_xor_data
from visual import plot_2d_datat
from sklearn.metrics import accuracy_score



X,y=generate_xor_data(n=200)

plot_2d_datat(X,y,title="XOR Data (Original Space)")


#linear model witout feature transformation

linear_model=LogisticRegression()
linear_model.fit(X,y)
y_pred_linear=linear_model.predict(X)
linear_accuracy=accuracy_score(y,y_pred_linear)
print(f"Accuracy of Linear Model on XOR Data without Feature Transformation: {linear_accuracy*100:.2f}%")
plot_2d_datat(X,y_pred_linear,title="XOR Data - Linear Model Predictions (Original Space)")
#Feature Transformation using Polynomial Features
poly=PolynomialFeatures(degree=2,include_bias=False)
X_poly=poly.fit_transform(X)
#Linear Model on Transformed Features
poly_model=LogisticRegression()
poly_model.fit(X_poly,y)
y_pred_poly=poly_model.predict(X_poly)
poly_accuracy=accuracy_score(y,y_pred_poly)
print(f"Accuracy of Linear Model on XOR Data with Feature Transformation: {poly_accuracy*100:.2f}%")
plot_2d_datat(X,y_pred_poly,title="XOR Data - Linear Model Predictions (Transformed Space)")