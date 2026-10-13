
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

target = "Goal_Difference"

# Use training data for exploratory analysis
# Keep final test matches separate
training_matches = int(len(df) * 0.80)
training_df = df.iloc[:training_matches]

# Calculate Pearson correlations
columns = predictors + [target]
correlation = training_df[columns].corr()

print("=== REGRESSION 2.1 CORRELATION ANALYSIS ===")
print(correlation.round(3).to_string())

# Save correlation table
correlation.to_csv("correlation_results_2_1.csv")

# Create heatmap
fig, ax = plt.subplots(figsize=(11, 9))

heatmap = ax.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

# Add variable names
ax.set_xticks(np.arange(len(columns)))
ax.set_yticks(np.arange(len(columns)))

ax.set_xticklabels(columns, rotation=45, ha="right")
ax.set_yticklabels(columns)

# Display correlation values in each square
for i in range(len(columns)):
    for j in range(len(columns)):
        value = correlation.iloc[i, j]

        text_color = (
            "white" if abs(value) > 0.65 else "black"
        )

        ax.text(
            j, i,
            f"{value:.2f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=9
        )

# Add colour scale
fig.colorbar(
    heatmap,
    ax=ax,
    label="Pearson Correlation"
)

ax.set_title(
    "Regression 2.1: Correlation Heatmap",
    fontsize=14,
    pad=15
)

plt.tight_layout()

# Save graph
plt.savefig(
    "correlation_heatmap_2_1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved: correlation_results_2_1.csv")
print("Saved: correlation_heatmap_2_1.png")
