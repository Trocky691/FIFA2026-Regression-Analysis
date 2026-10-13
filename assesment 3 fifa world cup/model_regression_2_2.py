
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load Regression 2.2 dataset
df = pd.read_csv("data/regression_2_2.csv")

# Exactly 8 pre-match explanatory variables
predictors = [
    "Team_PreMatch_PPG",
    "Opponent_PreMatch_PPG",
    "Team_Avg_GF",
    "Opponent_Avg_GF",
    "Team_Avg_GA",
    "Opponent_Avg_GA",
    "Team_FIFA_Rank",
    "Opponent_FIFA_Rank"
]

X = df[predictors]
y = df["Goals_Scored"]

# 104 matches, with 2 rows per match
assert len(df) == 208, "Expected 208 team-match records"
assert len(predictors) == 8

# Chronological 80/20 split by match
# Both teams from the same match stay together
total_matches = len(df) // 2
training_matches = int(total_matches * 0.80)
split_index = training_matches * 2

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Build Linear Regression model
model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

# Train on earlier matches
model.fit(X_train, y_train)

# Predict goals scored in later matches
predictions = model.predict(X_test)

# Calculate performance
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("=== REGRESSION 2.2 RESULTS ===")
print("Training matches:", training_matches)
print("Testing matches:", total_matches - training_matches)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\n=== MODEL PERFORMANCE ===")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2:", round(r2, 3))

# Save actual and predicted goals
results = pd.DataFrame({
    "Actual_Goals": y_test.to_numpy(),
    "Predicted_Goals": predictions,
    "Residual": y_test.to_numpy() - predictions
})

results.to_csv(
    "predictions_regression_2_2.csv",
    index=False
)

print("\nSaved as: predictions_regression_2_2.csv")
