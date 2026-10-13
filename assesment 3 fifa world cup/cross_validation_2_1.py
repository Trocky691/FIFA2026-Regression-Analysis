
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("data/regression_2_1.csv")

# Exactly 8 explanatory variables
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

# Use only the first 80% for cross-validation.
# Keep the final 20% untouched as the test set.
split_index = int(len(df) * 0.80)
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

# Time-series cross-validation keeps matches in order
tscv = TimeSeriesSplit(n_splits=5)

models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Lasso Regression": make_pipeline(StandardScaler(), Lasso(alpha=0.1))
}

results = []

for name, model in models.items():
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        model.fit(X_fold_train, y_fold_train)
        predictions = model.predict(X_fold_val)

        results.append({
            "Model": name,
            "Fold": fold,
            "MAE": mean_absolute_error(y_fold_val, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_fold_val, predictions)),
            "R2": r2_score(y_fold_val, predictions)
        })

results_df = pd.DataFrame(results)

print("=== REGRESSION 2.1 CROSS-VALIDATION ===")
print(results_df.round(3).to_string(index=False))

print("\n=== AVERAGE PERFORMANCE ===")
summary = results_df.groupby("Model")[["MAE", "RMSE", "R2"]].mean()
print(summary.round(3).to_string())

results_df.to_csv("cross_validation_results_2_1.csv", index=False)
print("\nSaved as: cross_validation_results_2_1.csv")
