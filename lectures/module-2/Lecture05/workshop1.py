import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Veriyi oku
df = pd.read_csv("raw_data.csv")

# 2. Hedef değişken oluştur: discount > 0 ise 1, değilse 0
df['target'] = (df['discount'] > 0).astype(int)


# Sayısal sütunlardan bazıları
features = [
    'unit_price', 'units_in_stock', 'units_on_order',
    'reorder_level', 'unit_price-2', 'quantity', 'employee_id'
]

X = df[features]
y = df['target']


# %60 train, %20 val, %20 test böl
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
svc = SVC(probability=True, random_state=42)

rf.fit(X_train, y_train)
svc.fit(X_train, y_train)


rf_val_pred = rf.predict_proba(X_val)
svc_val_pred = svc.predict_proba(X_val)

blend_X_val = np.hstack((rf_val_pred, svc_val_pred))  # 2 model x 2 class = 4 feature

meta_learner = LogisticRegression()
meta_learner.fit(blend_X_val, y_val)

rf_test_pred = rf.predict_proba(X_test)
svc_test_pred = svc.predict_proba(X_test)
blend_X_test = np.hstack((rf_test_pred, svc_test_pred))

final_pred = meta_learner.predict(blend_X_test)
print("Blending Accuracy:", accuracy_score(y_test, final_pred))
