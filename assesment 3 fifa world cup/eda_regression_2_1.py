import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Load Regression 2.1 dataset
df = pd.read_csv("data/regression_2_1.csv")

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

# Basic descriptive statistics
print("=== DESCRIPTIVE STATISTICS ===")
print(df[predictors].describe().round(3))

# Correlation matrix
correlation = df[predictors].corr()

print("\n=== CORRELATION MATRIX ===")
print(correlation.round(3))

# Find highly correlated predictor pairs
print("\n=== HIGH CORRELATIONS (|r| >= 0.80) ===")

found = False

for i in range(len(predictors)):
    for j in range(i + 1, len(predictors)):

        r = correlation.loc[predictors[i], predictors[j]]

        if abs(r) >= 0.80:
            print(
                predictors[i],
                "<->",
                predictors[j],
                ":",
                round(r, 3)
            )
            found = True

if not found:
    print("No predictor pairs have |r| >= 0.80.")
    # VIF check for multicollinearity
X = add_constant(df[predictors])

vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print("\n=== VARIANCE INFLATION FACTOR (VIF) ===")
print(vif.round(3).to_string(index=False))