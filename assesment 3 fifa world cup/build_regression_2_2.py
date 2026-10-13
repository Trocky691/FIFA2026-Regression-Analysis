
import pandas as pd

# Load our existing Regression 2.1 dataset
df = pd.read_csv("data/regression_2_1.csv")
# Load actual match scores
matches = pd.read_csv("data/matches.csv")

# Add scores to Regression 2.1 data using Match ID
df = df.merge(
    matches[["match_id", "home_score", "away_score"]],
    left_on="Match_ID",
    right_on="match_id",
    how="left",
    validate="one_to_one"
)

# Check that all scores were matched correctly
if df[["home_score", "away_score"]].isnull().any().any():
    raise ValueError("Some match scores could not be matched!")
rows = []

for _, match in df.iterrows():

    # Home team's record
    rows.append({
        "Team_PreMatch_PPG": match["Home_PreMatch_PPG"],
        "Opponent_PreMatch_PPG": match["Away_PreMatch_PPG"],
        "Team_Avg_GF": match["Home_Avg_GF"],
        "Opponent_Avg_GF": match["Away_Avg_GF"],
        "Team_Avg_GA": match["Home_Avg_GA"],
        "Opponent_Avg_GA": match["Away_Avg_GA"],
        "Team_FIFA_Rank": match["Home_FIFA_Rank"],
        "Opponent_FIFA_Rank": match["Away_FIFA_Rank"],
        "Goals_Scored": match["home_score"],
        "Is_Home": 1
    })

    # Away team's record
    rows.append({
        "Team_PreMatch_PPG": match["Away_PreMatch_PPG"],
        "Opponent_PreMatch_PPG": match["Home_PreMatch_PPG"],
        "Team_Avg_GF": match["Away_Avg_GF"],
        "Opponent_Avg_GF": match["Home_Avg_GF"],
        "Team_Avg_GA": match["Away_Avg_GA"],
        "Opponent_Avg_GA": match["Home_Avg_GA"],
        "Team_FIFA_Rank": match["Away_FIFA_Rank"],
        "Opponent_FIFA_Rank": match["Home_FIFA_Rank"],
        "Goals_Scored": match["away_score"],
        "Is_Home": 0
    })

# Create the new dataset
regression_2_2 = pd.DataFrame(rows)

# Check the dataset
print("=== REGRESSION 2.2 DATASET ===")
print("Total records:", len(regression_2_2))
print("Unique matches:", len(df))
print("Missing values:", regression_2_2.isnull().sum().sum())

print("\nFirst 5 rows:")
print(regression_2_2.head())

# Save the dataset
regression_2_2.to_csv(
    "data/regression_2_2.csv",
    index=False
)

print("\nSaved as: data/regression_2_2.csv")
