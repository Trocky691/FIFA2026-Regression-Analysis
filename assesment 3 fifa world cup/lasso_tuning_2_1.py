
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load Regression 2.1 dataset
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

# Keep the final 20% separate for testing
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

# Scaling is fitted separately within each CV fold
pipeline = make_pipeline(
    StandardScaler(),
    Lasso(max_iter=10000)
)

# Alpha values to test
parameters = {
    "lasso__alpha": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
}

# Preserve chronological order
tscv = TimeSeriesSplit(n_splits=5)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameters,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    n_jobs=-1,
    refit=True
)

search.fit(X_train, y_train)

# Show the best alpha and validation score
print("=== LASSO TUNING 2.1 ===")
print("Best alpha:", search.best_params_["lasso__alpha"])
print("Best CV MAE:", round(-search.best_score_, 3))

# Compare every alpha
results = pd.DataFrame({
    "Alpha": search.cv_results_["param_lasso__alpha"].astype(float),
    "Mean_CV_MAE": -search.cv_results_["mean_test_score"],
    "Std_CV_MAE": search.cv_results_["std_test_score"]
})

results = results.sort_values("Mean_CV_MAE")

print("\n=== ALPHA COMPARISON ===")
print(results.round(3).to_string(index=False))

results.to_csv("lasso_tuning_results_2_1.csv", index=False)

print("\nSaved as: lasso_tuning_results_2_1.csv")
