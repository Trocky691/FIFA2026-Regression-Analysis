import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv("data/regression_2_1.csv")

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

# --------------------------------------------------
# Chronological 80/20 split
# Earlier matches = training
# Later matches = testing
# --------------------------------------------------

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("=== BASELINE LINEAR REGRESSION 2.1 ===")
print("Total matches:", len(df))
print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n=== TEST SET PERFORMANCE ===")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R²:", round(r2, 3))

# Model coefficients
coefficients = pd.DataFrame({
    "Variable": predictors,
    "Coefficient": model.coef_
})

coefficients["Coefficient"] = coefficients["Coefficient"].round(3)

print("\n=== MODEL COEFFICIENTS ===")
print(coefficients.to_string(index=False))

# Actual vs predicted values
results = df.iloc[split_index:][
    ["Match_ID", "Home_Team", "Away_Team", "Goal_Difference"]
].copy()

results["Predicted_Goal_Difference"] = predictions
results["Predicted_Goal_Difference"] = (
    results["Predicted_Goal_Difference"].round(2)
)

print("\n=== SAMPLE PREDICTIONS ===")
print(results.head(10).to_string(index=False))