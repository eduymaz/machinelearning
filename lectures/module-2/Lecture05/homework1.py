# PROJE |  Northwind Sipariş İptal Olasılığı Tahmini
#  Librarys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# 1.  DATA 

df = pd.read_csv("hw.csv")

# 2. TARGET

df['order_date'] = pd.to_datetime(df['order_date'])
df['shipped_date'] = pd.to_datetime(df['shipped_date'], errors='coerce')
df['shipping_delay'] = (df['shipped_date'] - df['order_date']).dt.days
df['cancelled'] = ((df['shipped_date'].isna()) | (df['shipping_delay'] > 30)).astype(int)

# 3. Özellik mühendisliği

df['region'] = df['region'].fillna('Unknown')
df = pd.get_dummies(df, columns=['country', 'region'], drop_first=True)
features = ['product_count', 'total_order_amount', 'avg_product_price', 'avg_discount', 'shipping_delay']
one_hot_cols = [col for col in df.columns if col.startswith('country_') or col.startswith('region_')]
features += one_hot_cols
X = df[features]
y = df['cancelled']

# 4. Eğitim ve test seti ayrımı

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Eksik değer işlemleri

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 6. BLENDING

# Validation set ayır
X_train_base, X_val, y_train_base, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)

# Base modelleri oluştur ve eğit
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svc = SVC(probability=True, random_state=42)
rf.fit(X_train_base, y_train_base)
svc.fit(X_train_base, y_train_base)

# Validation set üzerinde tahminler (olasılık)
rf_val_pred = rf.predict_proba(X_val)[:, 1].reshape(-1, 1)
svc_val_pred = svc.predict_proba(X_val)[:, 1].reshape(-1, 1)
blend_X_val = np.hstack((rf_val_pred, svc_val_pred))

# Meta modeli eğit
meta = LogisticRegression()
meta.fit(blend_X_val, y_val)

# Test seti üzerinde base modellerin tahminlerini al
rf_test_pred = rf.predict_proba(X_test)[:, 1].reshape(-1, 1)
svc_test_pred = svc.predict_proba(X_test)[:, 1].reshape(-1, 1)
blend_X_test = np.hstack((rf_test_pred, svc_test_pred))

# Meta model ile test tahminleri
blend_pred = meta.predict(blend_X_test)
blend_pred_proba = meta.predict_proba(blend_X_test)[:, 1]

print("Blending Test Accuracy:", accuracy_score(y_test, blend_pred))
print("Blending Test ROC-AUC:", roc_auc_score(y_test, blend_pred_proba))

# 7. STACKING

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
stacking.fit(X_train, y_train)
stack_pred = stacking.predict(X_test)
stack_pred_proba = stacking.predict_proba(X_test)[:, 1]

print("Stacking Test Accuracy:", accuracy_score(y_test, stack_pred))
print("Stacking Test ROC-AUC:", roc_auc_score(y_test, stack_pred_proba))
