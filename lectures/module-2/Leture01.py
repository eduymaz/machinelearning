import pandas as pd
import numpy as np

np.random.seed(42)


df = pd.DataFrame(
    {
        "product_name" : np.random.choice(["chai", "chang", "tofu"], size=10),
        "shipper_company_name" : np.random.choice(["Yurtiçi", "Aras", "Sürat"], size=10),
        "unit_price" : np.round(np.random.uniform(10, 30, size=10), 2),
        "quantity" : np.random.randint(1, 10, size=10)
    }
)

df["total_price"] = df["unit_price"] * df["quantity"]

df["product_shipper_cross"] = df["product_name"]+"_"+df["shipper_company_name"]

#for encoding
df_encoded = pd.get_dummies(df[["product_shipper_cross"]], prefix="cross") # for prefix eklenebilir 

df_model = pd.concat([df, df_encoded], axis=1) #sütun bazlı birleştirme axis=1 


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = df_encoded
y = df_model["total_price"]

X_train,X_test,y_train_y_test = train_test_split(X,y,random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_prediction = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_prediction))

print(rmse)

# Task 1:  Kodu Northwind veri tabanına taşıyın
# Task 2:  XGBoost ve Lineer Regression ile de karşılaştırın 
