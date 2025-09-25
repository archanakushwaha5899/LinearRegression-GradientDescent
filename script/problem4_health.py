import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# -------------------- Linear Regression Class
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, iterations=2000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = 0
        self.c = 0
        self.loss_history = []

    def fit(self, X, y):
        n = len(y)
        for _ in range(self.iterations):
            y_pred = self.m * X + self.c
            error = y - y_pred
            dm = -(2/n) * np.sum(X * error)
            dc = -(2/n) * np.sum(error)
            self.m -= self.learning_rate * dm
            self.c -= self.learning_rate * dc
            self.loss_history.append(np.mean(error**2))

    def predict(self, X):
        return self.m * X + self.c

# ----------------------- Dataset generation
np.random.seed(0)
X = np.random.randint(20, 81, 300)                    # Age of patients (20–80)
y = 3 + 0.3 * X + np.random.normal(0, 2, 300)        # Recovery days (3–30) with noise

df = pd.DataFrame({"Age (years)": X, "Recovery Days": y.round(1)})
print("---- Healthcare Dataset (first 20 rows) ----")
print(df.head(20))

# -------------------- Create outputs folder
os.makedirs("outputs", exist_ok=True)

# -------------------- Scaling (important to prevent overflow)
X_scaled = (X - X.mean()) / X.std()
y_scaled = (y - y.mean()) / y.std()

# ------------------ Train model on scaled data
model = LinearRegressionGD(learning_rate=0.01, iterations=2000)
model.fit(X_scaled, y_scaled)

# Predictions (scaled)
y_pred_scaled = model.predict(X_scaled)

# Convert predictions back to original scale
y_pred = y_pred_scaled * y.std() + y.mean()

# --------------- Evaluation
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("---- Healthcare Problem ----")
print(f"Equation (scaled): y = {model.m:.4f}x + {model.c:.4f}")
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# --------------- Scatter plot + regression line
plt.figure(figsize=(6,4))
plt.scatter(X, y, color="blue", alpha=0.5, label="Actual")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("Age (years)")
plt.ylabel("Recovery Days")
plt.title("Healthcare: Age vs Recovery Days")
plt.legend()
plt.savefig("outputs/problem4_scatter.png")
plt.close()

# ---------- Loss curve
plt.figure(figsize=(6,4))
plt.plot(range(len(model.loss_history)), model.loss_history, color="green")
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Healthcare: Loss Curve")
plt.savefig("outputs/problem4_loss.png")
plt.close()
