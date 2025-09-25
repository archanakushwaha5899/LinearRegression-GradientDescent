import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------- Create outputs folder if not exist
os.makedirs("outputs", exist_ok=True)

# ------------------- Gradient Descent Linear Regression Class
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = 0
        self.c = 0
        self.loss_history = []

    def fit(self, X, y):
        n = float(len(X))
        for _ in range(self.iterations):
            y_pred = self.m * X + self.c
            error = y - y_pred
            d_m = (-2/n) * sum(X * error)
            d_c = (-2/n) * sum(error)
            self.m -= self.learning_rate * d_m
            self.c -= self.learning_rate * d_c
            self.loss_history.append(np.mean(error**2))

    def predict(self, X):
        return self.m * X + self.c


# ------------------- Problem 5: Logistics – Delivery Time vs Distance
np.random.seed(42)  # repeatability ke liye
X = np.random.randint(10, 200, 100)                # Distance (10–200 km)
y = 0.05 * X + np.random.normal(0, 2, 100)        # Delivery Time (hrs) with noise

df = pd.DataFrame({"Distance (km)": X, "Delivery Time (hrs)": y.round(2)})
print("---- Logistics Dataset (first 10 rows) ----")
print(df.head(10))

# ------------------- Scaling (to prevent overflow)
X_scaled = (X - X.mean()) / X.std()
y_scaled = (y - y.mean()) / y.std()

# ------------------- Train model
model = LinearRegressionGD(learning_rate=0.01, iterations=1000)
model.fit(X_scaled, y_scaled)

# Predictions (scaled)
y_pred_scaled = model.predict(X_scaled)

# Reverse scaling to original units
y_pred = (y_pred_scaled * y.std()) + y.mean()

# ------------------- Evaluation
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\n---- Logistics Problem ----")
print(f"Equation (scaled): y = {model.m:.4f}x + {model.c:.4f}")
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# ------------------- Scatter Plot with Regression Line
plt.figure(figsize=(6,4))
plt.scatter(X, y, color="blue", alpha=0.6, label="Actual")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (hrs)")
plt.title("Logistics: Distance vs Delivery Time")
plt.legend()
plt.savefig("outputs/problem5_logistics_scatter.png")
plt.close()

# ------------------- Loss Curve
plt.figure(figsize=(6,4))
plt.plot(range(len(model.loss_history)), model.loss_history, color="green")
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Logistics: Loss Curve")
plt.savefig("outputs/problem5_logistics_loss.png")
plt.close()
