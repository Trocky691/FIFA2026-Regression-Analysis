
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
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

# Chronological 80/20 split by match
# Keep both teams from each match together
training_matches = int((len(df) // 2) * 0.80)
split_index = training_matches * 2

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Compare three regression models
models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),
    "Ridge Regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),
    "Lasso Regression": make_pipeline(
        StandardScaler(),
        Lasso(alpha=0.1, max_iter=10000)
    )
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3)
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE")

print("=== REGRESSION 2.2 MODEL COMPARISON ===")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print()
print(results_df.to_string(index=False))

results_df.to_csv(
    "model_comparison_2_2.csv",
    index=False
)

print("\nSaved as: model_comparison_2_2.csv")
