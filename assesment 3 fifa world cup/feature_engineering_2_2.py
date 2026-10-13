
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("data/regression_2_2.csv")

# Original 8 variables
original_features = [
    "Team_PreMatch_PPG",
    "Opponent_PreMatch_PPG",
    "Team_Avg_GF",
    "Opponent_Avg_GF",
    "Team_Avg_GA",
    "Opponent_Avg_GA",
    "Team_FIFA_Rank",
    "Opponent_FIFA_Rank"
]

# Create new pre-match features
df["Rank_Difference"] = (
    df["Opponent_FIFA_Rank"] - df["Team_FIFA_Rank"]
)

df["PPG_Difference"] = (
    df["Team_PreMatch_PPG"] -
    df["Opponent_PreMatch_PPG"]
)

df["Attack_vs_Defence"] = (
    df["Team_Avg_GF"] -
    df["Opponent_Avg_GA"]
)

df["Defence_vs_Attack"] = (
    df["Opponent_Avg_GF"] -
    df["Team_Avg_GA"]
)

# Alternative set of exactly 8 features
engineered_features = [
    "Team_FIFA_Rank",
    "Opponent_FIFA_Rank",
    "Rank_Difference",
    "PPG_Difference",
    "Attack_vs_Defence",
    "Defence_vs_Attack",
    "Team_Avg_GF",
    "Opponent_Avg_GA"
]

# Use only the first 83 matches
total_matches = len(df) // 2
training_matches = int(total_matches * 0.80)

training_df = df.iloc[:training_matches * 2].copy()

# Five chronological validation folds
tscv = TimeSeriesSplit(n_splits=5)

results = []

feature_sets = {
    "Original Features": original_features,
    "Engineered Features": engineered_features
}

for feature_set_name, features in feature_sets.items():

    fold_maes = []

    for train_matches, val_matches in tscv.split(
        np.arange(training_matches)
    ):

        # Keep both team records together
        train_indices = np.concatenate([
            [2 * i, 2 * i + 1] for i in train_matches
        ])

        val_indices = np.concatenate([
            [2 * i, 2 * i + 1] for i in val_matches
        ])

        X_train = training_df.iloc[train_indices][features]
        y_train = training_df.iloc[train_indices]["Goals_Scored"]

        X_val = training_df.iloc[val_indices][features]
        y_val = training_df.iloc[val_indices]["Goals_Scored"]

        model = make_pipeline(
            StandardScaler(),
            Lasso(alpha=0.05, max_iter=10000)
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)
        fold_maes.append(mae)

    results.append({
        "Feature_Set": feature_set_name,
        "Mean_CV_MAE": np.mean(fold_maes),
        "Std_CV_MAE": np.std(fold_maes)
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Mean_CV_MAE")

print("=== REGRESSION 2.2 FEATURE ENGINEERING ===")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    "feature_engineering_results_2_2.csv",
    index=False
)

print("\nSaved: feature_engineering_results_2_2.csv")
