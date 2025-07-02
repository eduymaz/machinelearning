import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


from sklearn.ensemble import StackingClassifier



## BLENDING

X, y = make_classification(n_samples=1000, n_features=15, n_informative=5, random_state=42)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # %60 train, %20 val, %20 test

rf = RandomForestClassifier(n_estimators=100, random_state=42)
svc = SVC(probability=True, random_state=42)

rf.fit(X_train, y_train)
svc.fit(X_train, y_train)

rf_val_pred = rf.predict_proba(X_val)
svc_val_pred = svc.predict_proba(X_val)

blend_X_val = np.hstack((rf_val_pred, svc_val_pred))

meta_learner = LogisticRegression()
meta_learner.fit(blend_X_val, y_val)

rf_test_pred = rf.predict_proba(X_test)
svc_test_pred = svc.predict_proba(X_test)
blend_X_test = np.hstack((rf_test_pred, svc_test_pred))

blend_pred = meta_learner.predict(blend_X_test)

print("Blending Accuracy:", accuracy_score(y_test, blend_pred))