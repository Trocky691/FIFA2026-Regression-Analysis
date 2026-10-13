
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Regression 2.1: Goal Difference
df1 = pd.read_csv("data/regression_2_1.csv")

features1 = [
    "Home_PreMatch_PPG",
    "Away_PreMatch_PPG",
    "Home_Avg_GF",
    "Away_Avg_GF",
    "Home_Avg_GA",
    "Away_Avg_GA",
    "Home_FIFA_Rank",
    "Away_FIFA_Rank"
]

split1 = int(len(df1) * 0.8)

X1_train = df1.iloc[:split1][features1]
X1_test = df1.iloc[split1:][features1]

y1_train = df1.iloc[:split1]["Goal_Difference"]
y1_test = df1.iloc[split1:]["Goal_Difference"]

model1 = LinearRegression()
model1.fit(X1_train, y1_train)

pred1 = model1.predict(X1_test)

# Regression 2.2: Goals Scored
df2 = pd.read_csv("data/regression_2_2.csv")

features2 = [
    "Team_PreMatch_PPG",
    "Opponent_PreMatch_PPG",
    "Team_Avg_GF",
    "Opponent_Avg_GF",
    "Team_Avg_GA",
    "Opponent_Avg_GA",
    "Team_FIFA_Rank",
    "Opponent_FIFA_Rank"
]

# Keep both records of each match together
total_matches = len(df2) // 2
train_matches = int(total_matches * 0.8)
split2 = train_matches * 2

X2_train = df2.iloc[:split2][features2]
X2_test = df2.iloc[split2:][features2]

y2_train = df2.iloc[:split2]["Goals_Scored"]
y2_test = df2.iloc[split2:]["Goals_Scored"]

model2 = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

model2.fit(X2_train, y2_train)

pred2 = model2.predict(X2_test)

# Plot both models
plots = [
    (
        y1_test.to_numpy(),
        pred1,
        "Regression 2.1: Actual vs Predicted Goal Difference",
        "actual_vs_predicted_2_1.png"
    ),
    (
        y2_test.to_numpy(),
        pred2,
        "Regression 2.2: Actual vs Predicted Goals Scored",
        "actual_vs_predicted_2_2.png"
    )
]

for actual, predicted, title, filename in plots:

    plt.figure(figsize=(9, 6))

    plt.scatter(
        actual,
        predicted,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.4
    )

    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--",
        label="Perfect prediction"
    )

    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title(title)

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.show()

    print("Saved:", filename)

print("\nBoth graphs created successfully!")
