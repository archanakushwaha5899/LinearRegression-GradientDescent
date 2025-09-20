import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

class LinearRegressionGD:
    def __init__(self, learning_rate=0.000001, iterations=5000):
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

# -----------------------------
# Dataset generation
# -----------------------------
np.random.seed(0)
X = np.random.randint(20, 81, 300)
y = 2 + 0.2 * X + np.random.normal(0, 2, 300)

# Normalize X
X_norm = (X - X.mean()) / X.std()

os.makedirs("outputs", exist_ok=True)

# -----------------------------
# Train model
# -----------------------------
model = LinearRegressionGD(learning_rate=0.000001, iterations=5000)
model.fit(X_norm, y)

# Predictions
y_pred = model.predict(X_norm)

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("---- Healthcare Problem ----")
print(f"Equation: y = {model.m:.4f}*X_norm + {model.c:.4f}")
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# -----------------------------
# Scatter plot + regression line
# -----------------------------
plt.figure(figsize=(6,4))
plt.scatter(X, y, color="blue", alpha=0.5, label="Actual")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("Age (years)")
plt.ylabel("Recovery Days")
plt.title("Healthcare: Age vs Recovery Days")
plt.legend()
plt.savefig("outputs/problem4_scatter.png")
plt.close()

# -----------------------------
# Loss curve
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(range(len(model.loss_history)), model.loss_history, color="green")
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Healthcare: Loss Curve")
plt.savefig("outputs/problem4_loss.png")
plt.close()
