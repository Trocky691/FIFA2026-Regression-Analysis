
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Load Regression 2.1 dataset
df = pd.read_csv("data/regression_2_1.csv")

# Exactly 8 pre-match explanatory variables
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

# Chronological 80/20 split
training_matches = int(len(df) * 0.80)

X_train = X.iloc[:training_matches]
X_test = X.iloc[training_matches:]

y_train = y.iloc[:training_matches]
y_test = y.iloc[training_matches:]

# Use our final Linear Regression model
model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

model.fit(X_train, y_train)

# Standardised regression coefficients
coefficients = model.named_steps[
    "linearregression"
].coef_

# Measure feature importance on held-out data
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

print("=== REGRESSION 2.1 FEATURE IMPORTANCE ===")
print(results.round(4).to_string(index=False))

results.to_csv(
    "feature_importance_results_2_1.csv",
    index=False
)

# Create feature importance graph
plt.figure(figsize=(10, 6))

plt.barh(
    results["Feature"],
    results["Importance"],
    xerr=results["Importance_SD"],
    capsize=3
)

plt.axvline(
    0,
    color="black",
    linewidth=0.8
)

plt.xlabel("Increase in MAE when feature is shuffled")
plt.ylabel("Feature")
plt.title("Regression 2.1: Permutation Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    "feature_importance_2_1.png",
    dpi=300
)

plt.show()

print("\nSaved: feature_importance_results_2_1.csv")
print("Saved: feature_importance_2_1.png")
