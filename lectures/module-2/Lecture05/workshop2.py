import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# CSV'yi oku
df = pd.read_csv("raw_data.csv")

# Hedef: indirim var mı?
df['target'] = (df['discount'] > 0).astype(int)

# Sayısal feature'lar seçildi (dilersen daha fazlası eklenebilir)
features = [
    'unit_price', 'units_in_stock', 'units_on_order',
    'reorder_level', 'unit_price-2', 'quantity', 'employee_id'
]

X = df[features]
y = df['target']

# Train/Val/Test böl (60/20/20)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)


## BLENDING 

# Base modelleri eğit
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svc = SVC(probability=True, random_state=42)

rf.fit(X_train, y_train)
svc.fit(X_train, y_train)

# Validation set üzerindeki tahminler
rf_val_pred = rf.predict_proba(X_val)
svc_val_pred = svc.predict_proba(X_val)
blend_X_val = np.hstack((rf_val_pred, svc_val_pred))

# Meta model
meta_blend = LogisticRegression()
meta_blend.fit(blend_X_val, y_val)

# Test set tahminleri
rf_test_pred = rf.predict_proba(X_test)
svc_test_pred = svc.predict_proba(X_test)
blend_X_test = np.hstack((rf_test_pred, svc_test_pred))

blend_pred = meta_blend.predict(blend_X_test)
print("🔀 Blending Accuracy:", accuracy_score(y_test, blend_pred))


# STACKING 


# Base learners listesi
base_learners = [
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("svc", SVC(probability=True, random_state=42))
]

# Meta öğrenici
meta_stack = LogisticRegression()

# Stacking modeli
stack_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_stack,
    cv=5  # cross-validation stacking
)

stack_model.fit(X_train, y_train)
stack_pred = stack_model.predict(X_test)

print("📚 Stacking Accuracy:", accuracy_score(y_test, stack_pred))
