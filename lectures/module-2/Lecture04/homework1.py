# LIBRARYSssss
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import ElasticNet


# DATA 
df = pd.read_csv("data.csv")


# INDEPENDENTSSSS
X = df[['product_category', 'supplier', 'customer_country', 'employee_city', 'order_month', 'shipper_name']]

# TARGET 
y = df['total_amount']

# ONE HOT ENCODE

X_encoded = pd.get_dummies(X, drop_first=True)

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42)


# LASSO 

lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

y_pred_lasso = lasso.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f"Lasso MSE: {mse_lasso:.2f}")


coef_lasso = pd.Series(lasso.coef_, index=X_encoded.columns)
non_zero_coef = coef_lasso[coef_lasso != 0].sort_values(ascending=False)
print("Lasso modelde sıfır olmayan katsayılar:")
print(non_zero_coef)

#  LINEAR REG 

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)
print(f"Linear Regression MSE: {mse_linear:.2f}")


coef_linear = pd.Series(linear_model.coef_, index=X_encoded.columns)
print("Linear Regression katsayıları:")
print(coef_linear.sort_values(ascending=False))


# ELASTIC NET
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train, y_train)
y_pred_elastic = elastic.predict(X_test)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)
print(f"ElasticNet MSE: {mse_elastic:.2f}")

coef_elastic = pd.Series(elastic.coef_, index=X_encoded.columns)
non_zero_elastic = coef_elastic[coef_elastic != 0].sort_values(ascending=False)
print("ElasticNet modelde sıfır olmayan katsayılar:")
print(non_zero_elastic)

# SON KARŞILAŞTIRMA
print("\n--- MSE Karşılaştırması ---")
print(f"Linear Regression MSE: {mse_linear:.2f}")
print(f"Lasso MSE: {mse_lasso:.2f}")
print(f"ElasticNet MSE: {mse_elastic:.2f}")

