
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("data/regression_2_1.csv")

# Exactly 8 pre-match variables
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
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Compare baseline with tuned Lasso
models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),
    "Tuned Lasso (alpha=0.2)": make_pipeline(
        StandardScaler(),
        Lasso(alpha=0.2, max_iter=10000)
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

print("=== FINAL REGRESSION 2.1 COMPARISON ===")
print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

results_df.to_csv("final_model_results_2_1.csv", index=False)

print("\nSaved as: final_model_results_2_1.csv")
