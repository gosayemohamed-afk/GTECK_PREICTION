from collections import defaultdict, deque
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, log_loss

# Locate the CSV inside this project's data folder.
project_folder = Path(__file__).resolve().parent
csv_path = project_folder / "data" / "data_2025-2026.csv"

print("CSV path:", csv_path)
print("File found:", csv_path.exists())
with csv_path.open("r", encoding="utf-8-sig") as file:
    print("CSV header:", file.readline()[:400])
    print("First match row:", file.readline()[:400])

# Load the match data.
matches = pd.read_csv(csv_path)

# Display a few useful columns to check that the rows loaded correctly.
columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HC", "AC"]
print("Rows and columns:", matches.shape)
print(matches[columns].head(10).to_string(index=False))
from pathlib import Path
import pandas as pd

data_folder = Path(__file__).resolve().parent / "data"

seasons = [
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]

needed_columns = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HC", "AC"
]

season_tables = []

for season in seasons:
    csv_path = data_folder / f"data_{season}.csv"
    season_data = pd.read_csv(csv_path)

    # Keep only fields shared by all seasons.
    missing_columns = [
        column for column in needed_columns
        if column not in season_data.columns
    ]
    if missing_columns:
        raise ValueError(f"{season} is missing columns: {missing_columns}")

    season_data = season_data[needed_columns].copy()
    season_data["Date"] = pd.to_datetime(
        season_data["Date"], dayfirst=True, errors="coerce"
    )

    # Remove rows without a completed result or required match statistics.
    rows_before = len(season_data)
    season_data = season_data.dropna(subset=needed_columns).copy()
    print(f"{season}: {len(season_data)} complete matches from {rows_before} rows")

    season_data["Season"] = season
    season_tables.append(season_data)

# Join the seasons into one chronological table.
matches = pd.concat(season_tables, ignore_index=True)
matches = matches.sort_values("Date").reset_index(drop=True)

print("\nCombined rows and columns:", matches.shape)
print(matches.head().to_string(index=False))

from pathlib import Path
import pandas as pd

# Find this project's data folder, no matter where the script is run from.
project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"

# Use completed seasons only; skip the older file ending in "_bad".
season_files = [
    "data_2021-2022.csv",
    "data_2022-2023.csv",
    "data_2023-2024.csv",
    "data_2024-2025.csv",
    "data_2025-2026.csv",
]

frames = []

for filename in season_files:
    csv_path = data_folder / filename

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find: {csv_path}")

    season = pd.read_csv(csv_path, encoding="utf-8-sig")
    season["Season"] = filename.removeprefix("data_").removesuffix(".csv")
    frames.append(season)

matches = pd.concat(frames, ignore_index=True)

# These columns describe the fixture and its final result.
required_columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
missing_columns = [
    column for column in required_columns
    if column not in matches.columns
]

if missing_columns:
    raise ValueError(f"Required columns missing from the CSV files: {missing_columns}")

matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")

print("Combined seasons:", ", ".join(season_files))
print("Rows and columns:", matches.shape)
print("\nMatches per season:")
print(matches["Season"].value_counts().sort_index().to_string())

print("\nMissing values in important columns:")
print(matches[required_columns].isna().sum().to_string())

print("\nFirst five matches:")
print(
    matches[["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]]
    .head()
    .to_string(index=False)
)
print("\nCorner data check:")
if "HC" in matches.columns and "AC" in matches.columns:
    corner_rows = matches.dropna(subset=["HC", "AC"])
    print("Matches with both corner counts:", len(corner_rows), "of", len(matches))
    print("Matches missing corner counts:", len(matches) - len(corner_rows))
else:
    print("HC or AC column is missing from the season files.")
    # Make sure goal and corner columns are numeric.
for column in ["FTHG", "FTAG", "HC", "AC"]:
    matches[column] = pd.to_numeric(matches[column], errors="coerce")

matches = matches.sort_values("Date").reset_index(drop=True)

# Each team's history stores:
# goals scored, goals conceded, corners won, corners conceded.
team_history = defaultdict(lambda: deque(maxlen=5))
feature_rows = []

# Process one date at a time. This prevents a match's result from becoming
# a feature for another match played on that same date.
for match_date, games_on_date in matches.groupby("Date", sort=True):
    updates_for_after_today = []

    for _, game in games_on_date.iterrows():
        home_team = game["HomeTeam"]
        away_team = game["AwayTeam"]

        home_history = team_history[home_team]
        away_history = team_history[away_team]

        # Only make a training row once both teams have previous match data.
        if home_history and away_history:
            home_averages = [
                sum(values) / len(values)
                for values in zip(*home_history)
            ]
            away_averages = [
                sum(values) / len(values)
                for values in zip(*away_history)
            ]

            feature_rows.append({
                "Date": game["Date"],
                "Season": game["Season"],
                "HomeTeam": home_team,
                "AwayTeam": away_team,

                "Home_GoalsFor_Last5": home_averages[0],
                "Home_GoalsAgainst_Last5": home_averages[1],
                "Home_CornersFor_Last5": home_averages[2],
                "Home_CornersAgainst_Last5": home_averages[3],

                "Away_GoalsFor_Last5": away_averages[0],
                "Away_GoalsAgainst_Last5": away_averages[1],
                "Away_CornersFor_Last5": away_averages[2],
                "Away_CornersAgainst_Last5": away_averages[3],

                # These are the outcomes the models will learn to predict.
                "Result": game["FTR"],
                "HomeGoals": game["FTHG"],
                "AwayGoals": game["FTAG"],
                "HomeCorners": game["HC"],
                "AwayCorners": game["AC"],
            })

        # Save today's results for the teams' history, but add them only
        # after processing every match played today.
        updates_for_after_today.append(
            (home_team, (
                game["FTHG"], game["FTAG"], game["HC"], game["AC"]
            ))
        )
        updates_for_after_today.append(
            (away_team, (
                game["FTAG"], game["FTHG"], game["AC"], game["HC"]
            ))
        )

    for team, match_stats in updates_for_after_today:
        team_history[team].append(match_stats)

model_data = pd.DataFrame(feature_rows)

print("\nPre-match training data:")
print("Rows and columns:", model_data.shape)
print(model_data.head().to_string(index=False))
# Use only information from the teams' previous matches as model inputs.
feature_columns = [
    "Home_GoalsFor_Last5",
    "Home_GoalsAgainst_Last5",
    "Home_CornersFor_Last5",
    "Home_CornersAgainst_Last5",
    "Away_GoalsFor_Last5",
    "Away_GoalsAgainst_Last5",
    "Away_CornersFor_Last5",
    "Away_CornersAgainst_Last5",
]

# Train on earlier seasons and keep 2025-2026 aside for evaluation.
train_data = model_data[model_data["Season"] != "2025-2026"].copy()
test_data = model_data[model_data["Season"] == "2025-2026"].copy()

result_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    random_state=42,
)

result_model.fit(train_data[feature_columns], train_data["Result"])

predicted_results = result_model.predict(test_data[feature_columns])
result_probabilities = result_model.predict_proba(test_data[feature_columns])

print("\nMatch-result model evaluation")
print("Training matches:", len(train_data))
print("Test matches:", len(test_data))
print("Accuracy:", round(accuracy_score(test_data["Result"], predicted_results), 3))
print(
    classification_report(
        test_data["Result"],
        predicted_results,
        labels=result_model.classes_,
        target_names=[str(label) for label in result_model.classes_],
        zero_division=0,
    )
)

# Show each outcome probability for the first ten test matches.
predictions = test_data[["Date", "HomeTeam", "AwayTeam", "Result"]].copy()
predictions["PredictedResult"] = predicted_results

for index, label in enumerate(result_model.classes_):
    predictions[f"Probability_{label}"] = result_probabilities[:, index].round(3)

print("\nFirst ten test-match predictions:")
print(predictions.head(10).to_string(index=False))
regression_targets = {
    "HomeGoals": "PredictedHomeGoals",
    "AwayGoals": "PredictedAwayGoals",
    "HomeCorners": "PredictedHomeCorners",
    "AwayCorners": "PredictedAwayCorners",
}

score_predictions = test_data[
    ["Date", "HomeTeam", "AwayTeam",
     "HomeGoals", "AwayGoals", "HomeCorners", "AwayCorners"]
].copy()

for target, prediction_column in regression_targets.items():
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train_data[feature_columns], train_data[target])
    estimated_values = model.predict(test_data[feature_columns])
    score_predictions[prediction_column] = estimated_values

    error = mean_absolute_error(test_data[target], estimated_values)
    print(f"{target} mean absolute error: {error:.2f}")

score_predictions["PredictedScore"] = score_predictions.apply(
    lambda row: (
        f"{max(0, round(row['PredictedHomeGoals']))}-"
        f"{max(0, round(row['PredictedAwayGoals']))}"
    ),
    axis=1,
)

print("\nFirst ten score and corner predictions:")
print(score_predictions.head(10).round(2).to_string(index=False))
# Start with the result probabilities and predicted score.
market_predictions = predictions[
    [
        "Date", "HomeTeam", "AwayTeam", "PredictedResult",
        "Probability_H", "Probability_D", "Probability_A",
    ]
].copy().reset_index(drop=True)

market_predictions["PredictedScore"] = (
    score_predictions["PredictedScore"].to_numpy()
)

# Double chance probabilities are sums of the 1X2 probabilities:
# 1X = home win or draw; X2 = draw or away win; 12 = either team wins.
market_predictions["Probability_1X"] = (
    market_predictions["Probability_H"] + market_predictions["Probability_D"]
)
market_predictions["Probability_X2"] = (
    market_predictions["Probability_D"] + market_predictions["Probability_A"]
)
market_predictions["Probability_12"] = (
    market_predictions["Probability_H"] + market_predictions["Probability_A"]
)

# These labels are based on the historical match outcomes.
def make_btts_label(data):
    return (
        (data["HomeGoals"] > 0) & (data["AwayGoals"] > 0)
    ).astype(int)

def make_btts_over_2_5_label(data):
    total_goals = data["HomeGoals"] + data["AwayGoals"]
    return (
        (data["HomeGoals"] > 0)
        & (data["AwayGoals"] > 0)
        & (total_goals >= 3)
    ).astype(int)

def make_home_win_over_1_5_label(data):
    total_goals = data["HomeGoals"] + data["AwayGoals"]
    return (
        (data["Result"] == "H") & (total_goals >= 2)
    ).astype(int)

binary_markets = {
    "BTTS": make_btts_label,
    "BTTS_and_Over2_5": make_btts_over_2_5_label,
    "HomeWin_and_Over1_5": make_home_win_over_1_5_label,
}

for market_name, label_function in binary_markets.items():
    training_labels = label_function(train_data)

    market_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    market_model.fit(train_data[feature_columns], training_labels)

    positive_class_index = list(market_model.classes_).index(1)
    market_probability = market_model.predict_proba(
        test_data[feature_columns]
    )[:, positive_class_index]

    market_predictions[f"Probability_{market_name}"] = market_probability
    market_predictions[f"Prediction_{market_name}"] = [
        "Yes" if probability >= 0.5 else "No"
        for probability in market_probability
    ]

print("\nDouble chance, BTTS, and combo estimates:")
print(market_predictions.head(10).round(3).to_string(index=False))
print("\nDouble chance, BTTS, and combo estimates:")

display_predictions = market_predictions.head(10).copy()
numeric_columns = display_predictions.select_dtypes(include="number").columns
display_predictions[numeric_columns] = display_predictions[numeric_columns].round(3)

print(display_predictions.to_string(index=False))
models_to_compare = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    ),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    ),
}

baseline_result = train_data["Result"].mode().iloc[0]
baseline_accuracy = (test_data["Result"] == baseline_result).mean()

print(f"\nBaseline: always predict {baseline_result}")
print(f"Baseline accuracy: {baseline_accuracy:.3f}")

print("\nModel comparison:")
for name, model in models_to_compare.items():
    model.fit(train_data[feature_columns], train_data["Result"])

    predicted = model.predict(test_data[feature_columns])
    probabilities = model.predict_proba(test_data[feature_columns])

    print(
        f"{name}: "
        f"accuracy={accuracy_score(test_data['Result'], predicted):.3f}, "
        f"balanced_accuracy={balanced_accuracy_score(test_data['Result'], predicted):.3f}, "
        f"log_loss={log_loss(test_data['Result'], probabilities, labels=model.classes_):.3f}"
    )
    current_path = data_folder / "data_2026-2027.csv"
current_season = pd.read_csv(current_path, encoding="utf-8-sig")

print("\n2026-27 CSV check")
print("Rows:", len(current_season))
print("Columns:", current_season.columns.tolist()[:20])

preview_columns = [
    column for column in
    ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HC", "AC"]
    if column in current_season.columns
]
print(current_season[preview_columns].tail(12).to_string(index=False))

if all(column in current_season.columns for column in ["FTHG", "FTAG", "FTR"]):
    completed = current_season.dropna(subset=["FTHG", "FTAG", "FTR"])
    print("Played matches:", len(completed))
    print("Unplayed rows:", len(current_season) - len(completed))
    fixtures_path = data_folder / "fixtures.csv"
fixtures = pd.read_csv(fixtures_path, encoding="utf-8-sig")

print("\nUpcoming fixtures file")
print("File found:", fixtures_path.exists())
print("Columns:", fixtures.columns.tolist())
print("Number of fixtures:", len(fixtures))
print(fixtures.head(10).to_string(index=False))
from collections import defaultdict

# Combine the older seasons with completed matches from 2026–27.
history_matches = pd.concat([matches, current_season], ignore_index=True)
required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC"]
history_matches = history_matches[required].copy()

history_matches["Date"] = pd.to_datetime(
    history_matches["Date"], format="mixed", dayfirst=True, errors="coerce"
)

for column in ["FTHG", "FTAG", "HC", "AC"]:
    history_matches[column] = pd.to_numeric(history_matches[column], errors="coerce")

history_matches = history_matches.dropna(subset=required)
history_matches = history_matches.sort_values("Date")

# Store each team's goals and corners from its previous matches.
team_games = defaultdict(list)

for _, game in history_matches.iterrows():
    team_games[game["HomeTeam"]].append(
        (game["Date"], game["FTHG"], game["FTAG"], game["HC"], game["AC"])
    )
    team_games[game["AwayTeam"]].append(
        (game["Date"], game["FTAG"], game["FTHG"], game["AC"], game["HC"])
    )

# Load the upcoming games.
fixtures_path = data_folder / "epl_full_fixtures.csv"
raw_fixtures = pd.read_csv(fixtures_path, encoding="utf-8-sig")

# FixtureDownload leaves the result blank for matches that have not been played.
# Accept blank results and a dash so either CSV representation works.
result_text = raw_fixtures["Result"].fillna("").astype(str).str.strip()
upcoming = result_text.isin(["", "-"])
fixtures = raw_fixtures.loc[
    upcoming, ["Date", "Home Team", "Away Team"]
].copy()

fixtures = fixtures.rename(
    columns={"Home Team": "HomeTeam", "Away Team": "AwayTeam"}
)
fixtures["Date"] = pd.to_datetime(
    fixtures["Date"], dayfirst=True, errors="coerce"
)
fixtures = fixtures.dropna(subset=["Date"])
fixtures = fixtures[
    fixtures["Date"].dt.normalize() >= pd.Timestamp.today().normalize()
].copy()

# Match the names used in your historical data.
fixtures["HomeTeam"] = fixtures["HomeTeam"].replace(
    {"Man Utd": "Man United", "Spurs": "Tottenham"}
)
fixtures["AwayTeam"] = fixtures["AwayTeam"].replace(
    {"Man Utd": "Man United", "Spurs": "Tottenham"}
)

# Build features using only matches played before each fixture date.
future_rows = []

for _, fixture in fixtures.iterrows():
    home_games = [
        game for game in team_games[fixture["HomeTeam"]]
        if game[0] < fixture["Date"]
    ][-5:]

    away_games = [
        game for game in team_games[fixture["AwayTeam"]]
        if game[0] < fixture["Date"]
    ][-5:]

    if not home_games or not away_games:
        print(
            "Skipping fixture: not enough history for",
            fixture["HomeTeam"], "vs", fixture["AwayTeam"]
        )
        continue

    home_avg = [
        sum(game[i] for game in home_games) / len(home_games)
        for i in range(1, 5)
    ]
    away_avg = [
        sum(game[i] for game in away_games) / len(away_games)
        for i in range(1, 5)
    ]

    future_rows.append({
        "Date": fixture["Date"],
        "HomeTeam": fixture["HomeTeam"],
        "AwayTeam": fixture["AwayTeam"],
        "Home_GoalsFor_Last5": home_avg[0],
        "Home_GoalsAgainst_Last5": home_avg[1],
        "Home_CornersFor_Last5": home_avg[2],
        "Home_CornersAgainst_Last5": home_avg[3],
        "Away_GoalsFor_Last5": away_avg[0],
        "Away_GoalsAgainst_Last5": away_avg[1],
        "Away_CornersFor_Last5": away_avg[2],
        "Away_CornersAgainst_Last5": away_avg[3],
    })

future_data = pd.DataFrame(future_rows)

if future_data.empty:
    print("No fixtures had enough previous match data to make predictions.")
else:
    # Retrain on all past training rows, including 2025–26.
    result_model.fit(model_data[feature_columns], model_data["Result"])

    future_data["PredictedResult"] = result_model.predict(
        future_data[feature_columns]
    )

    probabilities = result_model.predict_proba(future_data[feature_columns])
    for index, label in enumerate(result_model.classes_):
        future_data[f"Probability_{label}"] = probabilities[:, index].round(3)

    print("\nUpcoming match result predictions:")
    print(
        future_data[
            ["Date", "HomeTeam", "AwayTeam", "PredictedResult"]
            + [f"Probability_{label}" for label in result_model.classes_]
        ].to_string(index=False)
    )
    # Train score and corner models using all historical training data.
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

score_targets = ["HomeGoals", "AwayGoals", "HomeCorners", "AwayCorners"]
missing_targets = [col for col in score_targets if col not in model_data.columns]

if future_data.empty:
    print("Skipping score and market predictions because there are no upcoming fixtures.")
elif missing_targets:
    print("Could not find these target columns:", missing_targets)
    print("Available columns:", model_data.columns.tolist())
else:
    score_models = {}

    for target in score_targets:
        score_model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        score_model.fit(model_data[feature_columns], model_data[target])
        score_models[target] = score_model
        future_data[f"Predicted{target}"] = score_model.predict(
            future_data[feature_columns]
        )

    # Turn predicted goal counts into a readable score, such as 2-1.
    future_data["PredictedScore"] = future_data.apply(
        lambda row: (
            f"{max(0, round(row['PredictedHomeGoals']))}-"
            f"{max(0, round(row['PredictedAwayGoals']))}"
        ),
        axis=1,
    )

    # Double chance probabilities are sums of the 1X2 probabilities.
    future_data["Probability_1X"] = (
        future_data["Probability_H"] + future_data["Probability_D"]
    )
    future_data["Probability_X2"] = (
        future_data["Probability_D"] + future_data["Probability_A"]
    )
    future_data["Probability_12"] = (
        future_data["Probability_H"] + future_data["Probability_A"]
    )

    # Create combo labels from the historical match results.
    historical = model_data.copy()
    total_goals = historical["HomeGoals"] + historical["AwayGoals"]
    both_score = (historical["HomeGoals"] > 0) & (historical["AwayGoals"] > 0)

    markets = {
        "BTTS": both_score.astype(int),
        "BTTS_Over2_5": (both_score & (total_goals > 2.5)).astype(int),
        "HomeWin_Over1_5": (
            (historical["Result"] == "H") & (total_goals > 1.5)
        ).astype(int),
    }

    for market_name, labels in markets.items():
        market_model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        market_model.fit(model_data[feature_columns], labels)

        positive_index = list(market_model.classes_).index(1)
        future_data[f"Probability_{market_name}"] = market_model.predict_proba(
            future_data[feature_columns]
        )[:, positive_index]

        future_data[f"Prediction_{market_name}"] = future_data[
            f"Probability_{market_name}"
        ].apply(lambda probability: "Yes" if probability >= 0.5 else "No")

    show_columns = [
        "Date", "HomeTeam", "AwayTeam", "PredictedResult", "PredictedScore",
        "PredictedHomeCorners", "PredictedAwayCorners",
        "Probability_1X", "Probability_X2", "Probability_12",
        "Probability_BTTS", "Prediction_BTTS",
        "Probability_BTTS_Over2_5", "Prediction_BTTS_Over2_5",
        "Probability_HomeWin_Over1_5", "Prediction_HomeWin_Over1_5",
    ]

    print("\nUpcoming match predictions: result, score, corners, and markets")
    print(future_data[show_columns].round(3).to_string(index=False))
