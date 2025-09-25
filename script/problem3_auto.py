import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# -------------------- Linear Regression Class
class LinearRegressionGD:
    def __init__(self, learning_rate=0.00001, iterations=2000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = 0
        self.c = 0
        self.loss_history = []

    def fit(self, X, y):
        n = len(y)
        self.m = 0
        self.c = 0
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
X = np.random.uniform(1.0, 3.5, 500)                     # Engine Size in liters
y = 50 - 8 * X + np.random.normal(0, 2, 500)            # Mileage km/l with noise

df = pd.DataFrame({"Engine Size (L)": X.round(2), "Mileage (km/l)": y.round(2)})

print("---- Automobile Dataset (first 20 rows) ----")
print(df.head(20))

# -------------------- Create outputs folder
os.makedirs("outputs", exist_ok=True)

# ------------------ Train model
model = LinearRegressionGD(learning_rate=0.001, iterations=2000)
model.fit(X, y)
y_pred = model.predict(X)

# --------------- Evaluation
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("---- Automobile Problem ----")
print(f"Equation: y = {model.m:.4f}x + {model.c:.4f}")
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# --------------- Scatter plot + regression line
plt.figure(figsize=(6,4))
plt.scatter(X, y, color="blue", alpha=0.5, label="Actual")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("Engine Size (L)")
plt.ylabel("Mileage (km/l)")
plt.title("Automobile: Engine Size vs Mileage")
plt.legend()
plt.savefig("outputs/problem3_scatter.png")
plt.close()

# ---------- Loss curve
plt.figure(figsize=(6,4))
plt.plot(range(len(model.loss_history)), model.loss_history, color="green")
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Automobile: Loss Curve")
plt.savefig("outputs/problem3_loss.png")
plt.close()
