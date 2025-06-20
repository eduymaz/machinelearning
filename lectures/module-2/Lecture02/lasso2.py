import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

df = pd.read_csv("tasks.csv")
print(df)

X = df.drop(["shipper_company_name", "product_name","product_shipper_cross", "total"], axis=1)
y = df["total"]

# L1 (Lasso)

lasso = LassoCV(cv=6)
lasso.fit(X,y)

feature_importance = pd.Series(lasso.coef_, index=X.columns)
print("Özelliklerin önem sıralaması:\n", feature_importance)