
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit
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

# Use only the first 83 matches (166 records)
# Keep the final 21 matches separate for testing
training_matches = int((len(df) // 2) * 0.80)
training_df = df.iloc[:training_matches * 2].copy()

# Give both team records the same match number
training_df["Match_Group"] = np.repeat(
    np.arange(training_matches), 2
)

# Cross-validation must split by match, not by row
match_indices = np.arange(training_matches)
tscv = TimeSeriesSplit(n_splits=5)

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

for model_name, model in models.items():

    for fold, (train_matches, val_matches) in enumerate(
        tscv.split(match_indices), start=1
    ):

        train_rows = training_df[
            training_df["Match_Group"].isin(train_matches)
        ]

        val_rows = training_df[
            training_df["Match_Group"].isin(val_matches)
        ]

        X_train = train_rows[predictors]
        y_train = train_rows["Goals_Scored"]

        X_val = val_rows[predictors]
        y_val = val_rows["Goals_Scored"]

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)
        rmse = np.sqrt(
            mean_squared_error(y_val, predictions)
        )
        r2 = r2_score(y_val, predictions)

        results.append({
            "Model": model_name,
            "Fold": fold,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

# Save all fold results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "cross_validation_results_2_2.csv",
    index=False
)

# Calculate average performance
summary = (
    results_df.groupby("Model")[["MAE", "RMSE", "R2"]]
    .mean()
    .sort_values("MAE")
)

print("=== REGRESSION 2.2 CROSS-VALIDATION ===")
print("Training matches:", training_matches)
print("Number of folds: 5")

print("\n=== AVERAGE PERFORMANCE ===")
print(summary.round(3).to_string())

print("\nSaved as: cross_validation_results_2_2.csv")
