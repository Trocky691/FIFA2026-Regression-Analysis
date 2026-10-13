
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Load Regression 2.2 dataset
df = pd.read_csv("data/regression_2_2.csv")

# Exactly 8 explanatory variables
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
training_matches = int((len(df) // 2) * 0.80)
split_index = training_matches * 2

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Use tuned Lasso from our earlier experiments
model = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.05, max_iter=10000)
)

model.fit(X_train, y_train)

# Standardised regression coefficients
coefficients = model.named_steps["lasso"].coef_

# Permutation importance on held-out data
importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=30,
    random_state=42
)

results = pd.DataFrame({
    "Feature": predictors,
    "Coefficient": coefficients,
    "Importance": importance.importances_mean,
    "Importance_SD": importance.importances_std
})

results = results.sort_values(
    "Importance",
    ascending=False
)

print("=== REGRESSION 2.2 FEATURE IMPORTANCE ===")
print(results.round(4).to_string(index=False))

results.to_csv(
    "feature_importance_results_2_2.csv",
    index=False
)

# Create a feature importance graph
plt.figure(figsize=(10, 6))

plt.barh(
    results["Feature"],
    results["Importance"],
    xerr=results["Importance_SD"],
    capsize=3
)

plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Increase in MAE when feature is shuffled")
plt.ylabel("Feature")
plt.title("Regression 2.2: Permutation Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    "feature_importance_2_2.png",
    dpi=300
)

plt.show()

print("\nSaved feature_importance_results_2_2.csv")
print("Saved feature_importance_2_2.png")
