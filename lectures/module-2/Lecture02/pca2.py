import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
df = pd.read_csv("tasks.csv")
print(df)

X = df.drop(["shipper_company_name", "product_name","product_shipper_cross", "total"], axis=1)
y = df["total"]


# L1 (Lasso)

lasso = LassoCV(cv=7)
lasso.fit(X,y)


# FOR SCALE ->>> PCA STEP 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA ->>> PCA STEP 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# ->>> PCA STEP 3
print("Açıklama varyans oranı: ", pca.explained_variance_ratio_)
print("Toplam açıklanan: ", pca.explained_variance_ratio_.sum())

# ->>> PCA STEP 4 // Visualization ******
loadings = pd.DataFrame(pca.components_.T, columns=["PCA1","PCA2"], index = X.columns)
print("Principical Components :\n", loadings)

feature_importance = pd.Series(lasso.coef_, index=X.columns)
print("Özellik Önem Sıralaması :\n", feature_importance)