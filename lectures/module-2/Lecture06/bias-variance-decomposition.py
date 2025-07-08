
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# Simülasyon (Northwind benzeri)
np.random.seed(42)
n_samples = 300

df = pd.DataFrame({
    "days_since_last_order": np.random.randint(0, 365, n_samples),
    "product_unit_price": np.random.uniform(5, 100, n_samples),
    "quantity": np.random.randint(1, 10, n_samples),
})

# Hedef değişken: Sipariş tutarı + gürültü
df["order_amount"] = (
    df["product_unit_price"] * df["quantity"]
    + 0.05 * df["days_since_last_order"]
    + np.random.normal(0, 10, n_samples)
)

X = df[["days_since_last_order", "product_unit_price", "quantity"]]
y = df["order_amount"]

# Test verisi (sabit)
X_test = X.sample(100, random_state=42).sort_values(by="days_since_last_order")
y_true = y.loc[X_test.index].values

from sklearn.metrics import mean_squared_error

def bias_variance_analysis_poly(degree, n_bootstrap=100):
    predictions = np.zeros((n_bootstrap, len(X_test)))

    for i in range(n_bootstrap):
        X_resampled, y_resampled = resample(X, y)
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)
        predictions[i] = y_pred

    avg_prediction = predictions.mean(axis=0)

    bias_squared = np.mean((avg_prediction - y_true) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    total_error = np.mean((predictions - y_true) ** 2)

    return bias_squared, variance, total_error


degrees = range(1, 7)
biases = []
variances = []
errors = []

for d in degrees:
    b, v, e = bias_variance_analysis_poly(d)
    biases.append(b)
    variances.append(v)
    errors.append(e)

# Görselleştirme
plt.figure(figsize=(10,6))
plt.plot(degrees, biases, label='Bias²', marker='o')
plt.plot(degrees, variances, label='Variance', marker='s')
plt.plot(degrees, errors, label='Total Error', marker='^')
plt.xlabel("Polynomial Degree")
plt.ylabel("Error")
plt.title("Northwind Tahmini için Bias-Variance Tradeoff")
plt.legend()
plt.grid(True)
plt.show()
