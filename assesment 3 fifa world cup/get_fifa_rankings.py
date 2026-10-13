import pandas as pd

df = pd.read_csv("data/matches.csv")

teams = sorted(
    set(df["home_team"]).union(set(df["away_team"]))
)

print("Total teams:", len(teams))
print("\n=== TEAM NAMES IN OUR DATASET ===")

for team in teams:
    print(team)
fifa_rank = {
    "Argentina": 1,
    "Spain": 2,
    "France": 3,
    "England": 4,
    "Portugal": 5,
    "Brazil": 6,
    "Morocco": 7,
    "Netherlands": 8,
    "Belgium": 9,
    "Germany": 10,
    "Croatia": 11,
    "Colombia": 13,
    "Mexico": 14,
    "Senegal": 15,
    "Uruguay": 16,
    "USA": 17,
    "Japan": 18,
    "Switzerland": 19,
    "Iran": 20,
    "Türkiye": 22,
    "Ecuador": 23,
    "Austria": 24,
    "Korea Republic": 25,
    "Australia": 27,
    "Algeria": 28,
    "Egypt": 29,
    "Canada": 30,
    "Norway": 31,
    "Côte d'Ivoire": 33,
    "Panama": 34,
    "Sweden": 38,
    "Czechia": 40,
    "Paraguay": 41,
    "Scotland": 42,
    "Tunisia": 45,
    "Congo DR": 46,
    "Uzbekistan": 50,
    "Qatar": 56,
    "Iraq": 57,
    "South Africa": 60,
    "Saudi Arabia": 61,
    "Jordan": 63,
    "Bosnia and Herzegovina": 64,
    "Cape Verde": 67,
    "Ghana": 73,
    "Curaçao": 82,
    "Haiti": 83,
    "New Zealand": 84,
    "Cabo Verde": 86,
    "IR Iran": 87,
}

missing = [team for team in teams if team not in fifa_rank]

print("\nTeams with missing FIFA ranking:", len(missing))
print(missing)