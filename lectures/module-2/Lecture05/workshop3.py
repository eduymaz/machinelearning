import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv("raw_data.csv")

# target -> discount 
df['discount'] = df['discount'].astype(float)
y = df['discount']


features = [
    'unit_price', 'units_in_stock', 'units_on_order',
    'reorder_level', 'unit_price-2', 'quantity', 'employee_id'
]
X = df[features]

# Eğitim, validation, test ->>>  (60/20/20)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
svr = SVR()

rf.fit(X_train, y_train)
svr.fit(X_train, y_train)

rf_val_pred = rf.predict(X_val).reshape(-1,1)
svr_val_pred = svr.predict(X_val).reshape(-1,1)
blend_X_val = np.hstack((rf_val_pred, svr_val_pred))

meta_blend = LinearRegression()
meta_blend.fit(blend_X_val, y_val)

# Test resutls : 
rf_test_pred = rf.predict(X_test).reshape(-1,1)
svr_test_pred = svr.predict(X_test).reshape(-1,1)
blend_X_test = np.hstack((rf_test_pred, svr_test_pred))

# Estimate and MSE
blend_pred = meta_blend.predict(blend_X_test)
print("🔀 Blending MSE:", mean_squared_error(y_test, blend_pred))

# StackingRegressor with base + meta model
base_learners = [
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("svr", SVR())
]

meta_stack = LinearRegression()

stack_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_stack,
    cv=5
)


stack_model.fit(X_train, y_train)
stack_pred = stack_model.predict(X_test)

# MSE
print("📚 Stacking MSE:", mean_squared_error(y_test, stack_pred))


