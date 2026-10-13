import pandas as pd

df = pd.read_csv("data/matches.csv")

print("=== FIFA WORLD CUP 2026 DATA VALIDATION ===")
print("Total matches:", len(df))
print("Unique match IDs:", df["match_id"].nunique())
print("Duplicate match IDs:", df["match_id"].duplicated().sum())
print("Missing home scores:", df["home_score"].isna().sum())
print("Missing away scores:", df["away_score"].isna().sum())
print("Total teams:", len(set(df["home_team"]) | set(df["away_team"])))