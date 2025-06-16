import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
#from sqlalchemy import create_engine
'''
engine = create_engine("postgresql+psycopg2://postgres:1@localhost:XXXX/Northwind")

query = """
SELECT
    p.product_name,
    s.company_name AS shipper_company_name,
    od.unit_price,
    od.quantity,
    od.unit_price * od.quantity AS total,
    p.product_name || '_' || s.company_name AS product_shipper_cross
FROM order_details od
JOIN products p ON od.product_id = p.product_id
JOIN orders o ON od.order_id = o.order_id
JOIN shippers s ON o.ship_via = s.shipper_id
"""
df = pd.read_sql(query, engine)
'''

df = pd.read_csv("tasks.csv")

# One-hot encoding
df_encoded = pd.get_dummies(df[["product_shipper_cross"]])

X = df_encoded
y = df["total"]

# Train-test bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# MODELS

# 1. Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

# 2. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

# 3. XGBoost
xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

# RESULTS 
print("📊 Model Karşılaştırma (RMSE):")
print(f"Random Forest RMSE       : {rf_rmse:.4f}")
print(f"Linear Regression RMSE   : {lr_rmse:.4f}")
print(f"XGBoost RMSE             : {xgb_rmse:.4f}")

# RESULTS
print("📊 Model Karşılaştırma (RMSE):")
print(f"Random Forest RMSE       : {rf_rmse:.4f}")
print(f"Linear Regression RMSE   : {lr_rmse:.4f}")
print(f"XGBoost RMSE             : {xgb_rmse:.4f}")


# RMSE COMPARING
results = pd.DataFrame({
    "Model": ["Random Forest", "Linear Regression", "XGBoost"],
    "RMSE": [rf_rmse, lr_rmse, xgb_rmse]
}).sort_values(by="RMSE")

print("\n📈 RMSE Karşılaştırma Tablosu:")
print(results.to_string(index=False))