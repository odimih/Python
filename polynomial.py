import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Synthetic data
np.random.seed(175)
n_samples = 20
X = np.sort(np.random.rand(n_samples)).reshape(-1, 1)
y = 2 * X.flatten() + 1 + np.random.normal(0, 0.2, n_samples)

# Prepare figure
fig, ax = plt.subplots(figsize=(8, 5))
print("X shape:", X.shape)
print("y shape:", y.shape)

ax.scatter(X, y, color="blue", label="data")
line, = ax.plot([], [], color="red", linewidth=2)
title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")

ax.set_xlim(0, 1)
ax.set_ylim(min(y)-0.5, max(y)+0.5)
ax.set_xlabel("X")
ax.set_ylabel("y")

# Smooth X for plotting
X_smooth = np.linspace(0, 1, 300).reshape(-1, 1)

def update(degree):
    poly = PolynomialFeatures(degree=degree)
    
    # Fit on original data
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict on smooth curve
    X_smooth_poly = poly.transform(X_smooth)
    y_smooth = model.predict(X_smooth_poly)

    # Update line
    line.set_data(X_smooth.flatten(), y_smooth.flatten())
    title.set_text(f"Polynomial degree = {degree}")
    return line, title

anim = FuncAnimation(fig, update, frames=range(1, 11), interval=800, blit=False)
plt.show()
