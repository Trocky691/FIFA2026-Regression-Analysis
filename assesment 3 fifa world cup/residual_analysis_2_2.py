
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load predictions from Regression 2.2
df = pd.read_csv("predictions_regression_2_2.csv")

actual = df["Actual_Goals"]
predicted = df["Predicted_Goals"]
residuals = actual - predicted

print("=== REGRESSION 2.2 RESIDUAL ANALYSIS ===")
print("Total predictions:", len(df))
print("Mean residual:", round(residuals.mean(), 3))
print("Residual standard deviation:", round(residuals.std(), 3))

# Graph 1: Residuals vs Predicted Goals
plt.figure(figsize=(8, 5))
plt.scatter(predicted, residuals, alpha=0.7)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Goals")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Regression 2.2: Residuals vs Predicted Goals")
plt.tight_layout()
plt.savefig("residual_plot_2_2.png", dpi=300)
plt.show()

# Graph 2: Residual Histogram
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=10, edgecolor="black")
plt.axvline(x=0, color="red", linestyle="--")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Regression 2.2: Residual Distribution")
plt.tight_layout()
plt.savefig("residual_histogram_2_2.png", dpi=300)
plt.show()

# Graph 3: Q-Q Plot
plt.figure(figsize=(8, 5))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Regression 2.2: Normal Q-Q Plot")
plt.tight_layout()
plt.savefig("residual_qq_plot_2_2.png", dpi=300)
plt.show()

print("\nGraphs saved successfully!")
