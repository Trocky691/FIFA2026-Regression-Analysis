import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load our regression dataset
df = pd.read_csv("data/regression_2_1.csv")

# Use the same 8 explanatory variables
predictors = [
    "Home_PreMatch_PPG",
    "Away_PreMatch_PPG",
    "Home_Avg_GF",
    "Away_Avg_GF",
    "Home_Avg_GA",
    "Away_Avg_GA",
    "Home_FIFA_Rank",
    "Away_FIFA_Rank"
]

X = df[predictors]
y = df["Goal_Difference"]

# Same chronological 80/20 split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict goal differences
predictions = model.predict(X_test)

# Calculate residuals
residuals = y_test - predictions

print("=== RESIDUAL ANALYSIS 2.1 ===")
print("Test matches:", len(y_test))
print("Mean residual:", round(residuals.mean(), 3))
print("Residual standard deviation:",
      round(residuals.std(), 3))

# Plot residuals against predictions
plt.figure(figsize=(8, 5))

plt.scatter(predictions, residuals, alpha=0.7)
plt.axhline(y=0, color="red", linestyle="--")

plt.xlabel("Predicted Goal Difference")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Regression 2.1 - Residual Analysis")

plt.tight_layout()
plt.savefig("residual_plot_2_1.png", dpi=300)
plt.show()
# Histogram of residuals
plt.figure(figsize=(8, 5))

plt.hist(residuals, bins=8, edgecolor="black")

plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Regression 2.1 - Residual Distribution")

plt.tight_layout()
plt.savefig("residual_histogram_2_1.png", dpi=300)
plt.show()
from scipy import stats

# Q-Q plot to check residual normality
plt.figure(figsize=(8, 5))

stats.probplot(residuals, dist="norm", plot=plt)

plt.title("Regression 2.1 - Residual Q-Q Plot")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")

plt.tight_layout()
plt.savefig("residual_qq_plot_2_1.png", dpi=300)
plt.show()