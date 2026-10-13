
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Load Regression 2.1 data
df = pd.read_csv("data/regression_2_1.csv")

# Original eight pre-match variables
original_features = [
    "Home_PreMatch_PPG",
    "Away_PreMatch_PPG",
    "Home_Avg_GF",
    "Away_Avg_GF",
    "Home_Avg_GA",
    "Away_Avg_GA",
    "Home_FIFA_Rank",
    "Away_FIFA_Rank"
]

# Create engineered features using pre-match data only
df["Rank_Difference"] = (
    df["Away_FIFA_Rank"] - df["Home_FIFA_Rank"]
)

df["PPG_Difference"] = (
    df["Home_PreMatch_PPG"] - df["Away_PreMatch_PPG"]
)

df["Attack_Difference"] = (
    df["Home_Avg_GF"] - df["Away_Avg_GF"]
)

df["Defence_Difference"] = (
    df["Away_Avg_GA"] - df["Home_Avg_GA"]
)

# Alternative eight-feature set
engineered_features = [
    "Rank_Difference",
    "PPG_Difference",
    "Attack_Difference",
    "Defence_Difference",
    "Home_FIFA_Rank",
    "Away_FIFA_Rank",
    "Home_Avg_GF",
    "Away_Avg_GA"
]

# Use the first 80% of matches for training
training_size = int(len(df) * 0.80)
training_df = df.iloc[:training_size].copy()

tscv = TimeSeriesSplit(n_splits=5)

feature_sets = {
    "Original Features": original_features,
    "Engineered Features": engineered_features
}

results = []

for name, features in feature_sets.items():

    fold_maes = []

    for train_idx, val_idx in tscv.split(training_df):

        X_train = training_df.iloc[train_idx][features]
        y_train = training_df.iloc[train_idx]["Goal_Difference"]

        X_val = training_df.iloc[val_idx][features]
        y_val = training_df.iloc[val_idx]["Goal_Difference"]

        model = make_pipeline(
            StandardScaler(),
            Lasso(alpha=0.2, max_iter=10000)
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)
        fold_maes.append(mae)

    results.append({
        "Feature_Set": name,
        "Mean_CV_MAE": np.mean(fold_maes),
        "Std_CV_MAE": np.std(fold_maes)
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Mean_CV_MAE")

print("=== REGRESSION 2.1 FEATURE ENGINEERING ===")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    "feature_engineering_results_2_1.csv",
    index=False
)

print("\nSaved: feature_engineering_results_2_1.csv")
