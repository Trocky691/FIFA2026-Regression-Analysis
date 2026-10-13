import pandas as pd
from get_fifa_rankings import fifa_rank as fifa_rankings
# Load the 104-match dataset
df = pd.read_csv("data/matches.csv")

# Sort matches chronologically
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "match_number"]).reset_index(drop=True)

# Store each team's previous tournament performance
team_history = {}

rows = []

def get_pre_match_stats(team):
    """Return statistics using ONLY matches played before the current match."""

    history = team_history.get(team, [])

    # Neutral starting values for a team's first match
    if len(history) == 0:
        return {
            "PPG": 0.0,
            "Avg_GF": 0.0,
            "Avg_GA": 0.0,
            "Matches_Played": 0
        }

    games = len(history)
    total_points = sum(game["points"] for game in history)
    goals_for = sum(game["gf"] for game in history)
    goals_against = sum(game["ga"] for game in history)

    return {
        "PPG": total_points / games,
        "Avg_GF": goals_for / games,
        "Avg_GA": goals_against / games,
        "Matches_Played": games
    }


for _, match in df.iterrows():

    home = match["home_team"]
    away = match["away_team"]

    home_goals = match["home_score"]
    away_goals = match["away_score"]

    # IMPORTANT:
    # Calculate predictors BEFORE adding the current match to history.
    home_stats = get_pre_match_stats(home)
    away_stats = get_pre_match_stats(away)

    rows.append({
        "Match_ID": match["match_id"],
        "Match_Number": match["match_number"],
        "Date": match["date"],
        "Home_Team": home,
        "Away_Team": away,

        # Exactly 8 explanatory variables
        "Home_PreMatch_PPG": home_stats["PPG"],
        "Away_PreMatch_PPG": away_stats["PPG"],
        "Home_Avg_GF": home_stats["Avg_GF"],
        "Away_Avg_GF": away_stats["Avg_GF"],
        "Home_Avg_GA": home_stats["Avg_GA"],
        "Away_Avg_GA": away_stats["Avg_GA"],
        "Home_FIFA_Rank": fifa_rankings[home],
        "Away_FIFA_Rank": fifa_rankings[away],

        # Target variable
        "Goal_Difference": home_goals - away_goals
    })

    # Determine points AFTER the match
    if home_goals > away_goals:
        home_points = 3
        away_points = 0
    elif home_goals < away_goals:
        home_points = 0
        away_points = 3
    else:
        home_points = 1
        away_points = 1

    # Now update history AFTER predictors were calculated
    team_history.setdefault(home, []).append({
        "gf": home_goals,
        "ga": away_goals,
        "points": home_points
    })

    team_history.setdefault(away, []).append({
        "gf": away_goals,
        "ga": home_goals,
        "points": away_points
    })


reg21 = pd.DataFrame(rows)

# Save the modelling dataset
reg21.to_csv("data/regression_2_1.csv", index=False)

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

print("=== REGRESSION 2.1 DATASET ===")
print("Rows:", len(reg21))
print("Number of explanatory variables:", len(predictors))
print("Missing predictor values:", reg21[predictors].isna().sum().sum())

print("\nExplanatory variables:")
for i, variable in enumerate(predictors, start=1):
    print(f"{i}. {variable}")

print("\nFirst 5 rows:")
print(reg21.head().to_string(index=False))

print("\nSaved as: data/regression_2_1.csv")