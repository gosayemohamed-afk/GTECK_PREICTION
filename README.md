# Gteck Prediction

A beginner data science project that estimates English Premier League match outcomes from historical match data.

## Predictions

- Home win, draw, and away win probabilities
- Estimated score and corner counts
- Double chance probabilities
- Both teams to score and selected goal combinations

These are statistical estimates for learning and analysis, not guaranteed outcomes.

## Run locally

1. Install Python.
2. Create and activate a virtual environment.
3. Install the packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Start the web app from the project folder:

   ```powershell
   python -m streamlit run app.py
   ```

The app reads match data and scheduled fixtures from the `data` folder.

## Project files

- `app.py` — Streamlit match-centre interface
- `football_prediction.py` — data preparation, model training, and predictions
- `data/` — historical match CSV files and `fixtures.csv`
- `requirements.txt` — Python packages needed to run the app

## Data

Match data is stored locally in the CSV files under `data/`. Check the data provider's terms before redistributing the CSV files publicly.
