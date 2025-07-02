#ürün fiyatı, kargo süresi, personel tecrübesi, müşteri şikayet sayısı, reklam harcaması, satış ekibi sayısı....

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet



np.random.seed(42)

n = 500
df = pd.DataFrame({
    "ad_budget": np.random.uniform(1000,5000,n), # 1000 5000
    "sales_team_size": np.random.randint(1000,2000,n), # 
    "delivery_days": np.random.randint(1000,1900,n)
})

# Çoklu Doğrusal Bağlantılı veri, Multicollinearity
df["marketing_expense"] = df["ad_budget"] + np.random.normal(0,200,n)

# ! NOT : Çoklu doğrusal bağlantı var ise katsayıların kararsız hale geldiği bir durum oluşur. Yorumlanabilirlik düşer.

df["sales"] = (
    df["ad_budget"]*0.5 + df["sales_team_size"]*0.3 + df["delivery_days"]*2 + np.random.normal(0,200,n)
)

X = df.drop("sales", axis=1)
y = df["sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


ridge = Ridge(alpha=100)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print("Ridge MSE:", mean_squared_error(y_test, y_pred_ridge))
print("Ridge Katsayılar", pd.Series(ridge.coef_, index=X.columns))

# LINEAR
linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred_linear = linear.predict(X_test)

print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_linear))
print("Linear Katsayılar", pd.Series(linear.coef_, index=X.columns))


# Lasso modeli
lasso = Lasso(alpha=200)  # alpha değeri Ridge'den farklı çalışır; genellikle daha küçük tutulur
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print("Lasso MSE:", mean_squared_error(y_test, y_pred_lasso))
print("Lasso Katsayılar:", pd.Series(lasso.coef_, index=X.columns))


# ELASTIC NET

elastic = ElasticNet(alpha=100, l1_ratio=0.5)
elastic.fit(X_train, y_train)
y_pred_elasttic = elastic.predict(X_test)

print("Elastic MSE:", mean_squared_error(y_test, y_pred_elasttic))
print("Elastic Katsayılar:", pd.Series(elastic.coef_, index=X.columns))
