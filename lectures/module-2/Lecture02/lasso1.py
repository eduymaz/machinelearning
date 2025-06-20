import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

np.random.seed()

df=pd.DataFrame({
    "product_price" : np.random.uniform(10,100,20000),
    "employee_experience_years" : np.random.randint(1,20,20000),
    "customer_order_count" : np.random.randint(1,50,20000),
    "shipper_delivery_time_days" : np.random.randint(1,10,20000),
    "order_total_amount" : np.random.uniform(100,1000,20000)
})

X = df.drop("order_total_amount", axis=1)
y = df["order_total_amount"]


# L1 (Lasso)

lasso = LassoCV(cv=7)
lasso.fit(X,y)

feature_importance = pd.Series(lasso.coef_, index=X.columns)
print("Özellik öenmleri :\n", feature_importance)