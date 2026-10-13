
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load Regression 2.2 dataset
df = pd.read_csv("data/regression_2_2.csv")

# Exactly 8 pre-match variables
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

# Keep only the first 83 matches for tuning
total_matches = len(df) // 2
training_matches = int(total_matches * 0.80)

training_df = df.iloc[:training_matches * 2]

X_train = training_df[predictors]
y_train = training_df["Goals_Scored"]

# Define Lasso model with scaling
pipeline = make_pipeline(
    StandardScaler(),
    Lasso(max_iter=10000)
)

# Alpha values to compare
parameters = {
    "lasso__alpha": [
        0.001, 0.01, 0.05, 0.1,
        0.2, 0.5, 1.0, 2.0
    ]
}

# Split by match so both teams stay together
match_splits = TimeSeriesSplit(n_splits=5)

cv_splits = []

for train_matches, val_matches in match_splits.split(
    np.arange(training_matches)
):
    train_rows = np.repeat(train_matches * 2, 2) + np.tile(
        [0, 1], len(train_matches)
    )

    val_rows = np.repeat(val_matches * 2, 2) + np.tile(
        [0, 1], len(val_matches)
    )

    cv_splits.append((train_rows, val_rows))

# Tune using training data only
search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameters,
    scoring="neg_mean_absolute_error",
    cv=cv_splits,
    n_jobs=-1,
    refit=True
)

search.fit(X_train, y_train)

print("=== LASSO TUNING 2.2 ===")
print("Best alpha:", search.best_params_["lasso__alpha"])
print("Best CV MAE:", round(-search.best_score_, 3))

# Display results for every alpha
results = pd.DataFrame({
    "Alpha": search.cv_results_["param_lasso__alpha"].astype(float),
    "Mean_CV_MAE": -search.cv_results_["mean_test_score"],
    "Std_CV_MAE": search.cv_results_["std_test_score"]
})

results = results.sort_values("Mean_CV_MAE")

print("\n=== ALPHA COMPARISON ===")
print(results.round(3).to_string(index=False))

results.to_csv(
    "lasso_tuning_results_2_2.csv",
    index=False
)

print("\nSaved as: lasso_tuning_results_2_2.csv")
