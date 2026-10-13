
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("data/regression_2_1.csv")

# Use the same 8 explanatory variables
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

# Models to compare
models = {
    "Linear Regression": LinearRegression(),
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

print("\n=== REGRESSION 2.1 MODEL COMPARISON ===")
print(results_df.to_string(index=False))

results_df.to_csv("model_comparison_2_1.csv", index=False)

print("\nSaved as: model_comparison_2_1.csv")
