import html
import importlib

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Gteck Prediction | Match predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --page: #06131a;
        --panel: #0c2029;
        --panel-soft: #102832;
        --line: rgba(181, 214, 222, .11);
        --muted: #8ea5ad;
        --white: #edf5f5;
        --pink: #fa315f;
        --mint: #4de0b3;
        --blue: #66aef7;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        color: var(--white);
        background:
          radial-gradient(ellipse at 75% -10%, rgba(24, 77, 88, .42), transparent 42%),
          var(--page);
    }
    [data-testid="stHeader"] { background: rgba(6, 19, 26, .75); }
    [data-testid="stSidebar"] {
        background: #081a21;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c3d3d7; }
    .block-container { max-width: 1420px; padding-top: 1.4rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--white); font-family: 'Space Grotesk', sans-serif; }
    .brand-row {
        display:flex; align-items:center; justify-content:space-between;
        padding: 3px 0 20px; border-bottom: 1px solid var(--line); margin-bottom: 24px;
    }
    .brand { font: 700 22px 'Space Grotesk',sans-serif; letter-spacing: 1.2px; color: var(--white); }
    .brand span { color: var(--pink); }
    .brand-caption { color: var(--muted); font-size: 12px; letter-spacing: 1.5px; text-transform: uppercase; }
    .eyebrow { color: var(--mint); font-size: 11px; font-weight: 800; letter-spacing: 1.7px; text-transform: uppercase; }
    .page-title { font: 700 32px 'Space Grotesk',sans-serif; margin: 5px 0 2px; color: var(--white); }
    .page-subtitle { color: var(--muted); font-size: 14px; margin-bottom: 22px; }
    .section-title { font: 700 17px 'Space Grotesk',sans-serif; margin: 22px 0 12px; color: var(--white); }
    .section-note { color: var(--muted); font-size: 12px; margin-top: -8px; margin-bottom: 14px; }
    .match-card {
        background: linear-gradient(135deg, rgba(17, 43, 53, .98), rgba(10, 29, 38, .98));
        border: 1px solid var(--line); border-radius: 18px; padding: 22px 24px;
        box-shadow: 0 18px 48px rgba(0,0,0,.18); margin: 8px 0 16px;
    }
    .match-meta { display:flex; justify-content:space-between; align-items:center; gap:12px; color:var(--muted); font-size:12px; }
    .league-pill { color:#dce8e9; background:#18343d; border-radius:99px; padding:6px 10px; font-size:11px; font-weight:700; letter-spacing:.5px; }
    .teams {
        display:grid; grid-template-columns:1fr auto 1fr; align-items:center;
        gap:15px; padding:26px 0 22px;
    }
    .team { display:flex; align-items:center; gap:13px; min-width:0; }
    .team.away { justify-content:flex-end; text-align:right; }
    .team-badge {
        display:flex; flex:0 0 46px; height:46px; align-items:center; justify-content:center;
        border-radius:50%; color:#dff7f1; font:700 13px 'Space Grotesk',sans-serif;
        background:linear-gradient(145deg,#17424a,#102b34); border:1px solid rgba(116,218,195,.28);
    }
    .team-name { color:var(--white); font:700 clamp(15px,2vw,21px) 'Space Grotesk',sans-serif; overflow-wrap:anywhere; }
    .vs { color:#6f8790; font-size:11px; font-weight:800; letter-spacing:1px; }
    .prediction-line { display:flex; justify-content:center; align-items:center; gap:12px; color:#bdd0d4; font-size:12px; }
    .score-pill { background:#07171d; color:var(--white); border:1px solid var(--line); border-radius:10px; padding:8px 16px; font:700 20px 'Space Grotesk',sans-serif; }
    .pred-tag { color:var(--mint); font-weight:800; background:rgba(77,224,179,.10); border:1px solid rgba(77,224,179,.18); border-radius:99px; padding:7px 11px; }
    .prob-card, .market-card {
        background:var(--panel); border:1px solid var(--line); border-radius:14px;
        padding:15px 16px; height:100%;
    }
    .prob-head { display:flex; justify-content:space-between; color:#cad9dc; font-size:12px; font-weight:600; margin-bottom:10px; }
    .prob-value { color:var(--white); font:700 18px 'Space Grotesk',sans-serif; }
    .bar-track { height:6px; border-radius:9px; background:#1b343d; overflow:hidden; }
    .bar-fill { height:100%; border-radius:9px; }
    .market-label { min-height:36px; color:#bbccd0; font-size:12px; line-height:1.45; }
    .market-value { color:var(--white); font:700 21px 'Space Grotesk',sans-serif; margin:7px 0 2px; }
    .market-status { color:var(--mint); font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.8px; }
    .schedule-card {
        display:grid; grid-template-columns:100px minmax(100px,1fr) 38px minmax(100px,1fr) 70px;
        align-items:center; gap:10px; padding:13px 16px; margin:7px 0;
        background:rgba(12,32,41,.78); border:1px solid var(--line); border-radius:12px;
    }
    .schedule-date { color:var(--muted); font-size:11px; }
    .schedule-team { color:#e0ebed; font-size:13px; font-weight:700; }
    .schedule-team.right { text-align:right; }
    .schedule-vs { color:#708891; text-align:center; font-size:10px; font-weight:800; }
    .schedule-score { justify-self:end; color:var(--mint); font:700 13px 'Space Grotesk',sans-serif; }
    div[data-testid="stSelectbox"] label, div[data-testid="stDateInput"] label { color:#b9cbd0; font-size:12px; }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stDateInput"] input { background:#102832; border-color:var(--line); color:var(--white); border-radius:10px; }
    div[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); padding:12px 14px; border-radius:13px; }
    div[data-testid="stMetricLabel"] { color:var(--muted); }
    div[data-testid="stMetricValue"] { color:var(--white); font-family:'Space Grotesk',sans-serif; }
    hr { border-color:var(--line); }
    .footnote { color:#728891; font-size:11px; margin-top:20px; }
    @media(max-width:700px) {
      .block-container { padding-left:1rem; padding-right:1rem; }
      .schedule-card { grid-template-columns:74px 1fr 25px 1fr 48px; gap:5px; padding:11px 8px; }
      .schedule-team { font-size:11px; }
      .schedule-date { font-size:10px; }
      .brand-caption { display:none; }
      .match-card { padding:17px 14px; }
      .team-badge { flex-basis:36px; height:36px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictions():
    # Reload the model script so deployments use the latest fixtures and code.
    import football_prediction
    importlib.reload(football_prediction)

    return football_prediction.future_data.copy().reset_index(drop=True)


def safe_text(value):
    return html.escape(str(value))


def probability(value):
    try:
        if pd.isna(value):
            return None
        number = float(value)
        # Probabilities are stored from 0 to 1. Clamp only for display safety.
        return max(0.0, min(1.0, number))
    except (TypeError, ValueError):
        return None


def display_percent(value):
    value = probability(value)
    return "—" if value is None else f"{value:.1%}"


def initials(team):
    words = [word for word in str(team).replace("'", "").split() if word]
    if not words:
        return "FC"
    if len(words) == 1:
        return words[0][:2].upper()
    return "".join(word[0] for word in words[:2]).upper()


def render_probability_card(label, value, color):
    amount = probability(value)
    width = 0 if amount is None else amount * 100
    shown = "—" if amount is None else f"{amount:.1%}"
    st.markdown(
        f"""
        <div class="prob-card">
          <div class="prob-head"><span>{safe_text(label)}</span><span class="prob-value">{shown}</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;background:{color};"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_card(label, value, prediction=None):
    status = "<div class='market-status'>Model leans yes</div>" if prediction == "Yes" else (
        "<div class='market-status' style='color:#9fb1b6'>Model leans no</div>" if prediction == "No" else ""
    )
    st.markdown(
        f"""
        <div class="market-card">
          <div class="market-label">{safe_text(label)}</div>
          <div class="market-value">{display_percent(value)}</div>
          {status}
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    predictions = load_predictions()

    st.markdown(
        """
        <div class="brand-row">
        <div class="brand">GTECK <span>PREDICTION</span></div>
          <div class="brand-caption">Premier League · Match predictions</div>
        </div>
        <div class="eyebrow">The match centre</div>
        <div class="page-title">Upcoming fixtures</div>
        <div class="page-subtitle">Explore model estimates for results, scores, corners, and goal markets.</div>
        """,
        unsafe_allow_html=True,
    )

    if predictions.empty:
        st.warning("No predictions are available. Check your fixtures.csv file.")
    else:
        predictions["Date"] = pd.to_datetime(predictions["Date"], errors="coerce")
        predictions = predictions.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        if predictions.empty:
            st.warning("The fixtures file has no readable match dates.")
        else:
            with st.sidebar:
                st.markdown("### ⚽ Match finder")
                st.caption("Choose a matchday and fixture to view its model estimates.")
                date_values = predictions["Date"].dt.strftime("%Y-%m-%d").drop_duplicates().tolist()
                date_filter = st.selectbox("Matchday", ["All matchdays", *date_values])

                if date_filter == "All matchdays":
                    available = predictions
                else:
                    available = predictions[
                        predictions["Date"].dt.strftime("%Y-%m-%d") == date_filter
                    ]

                fixture_indices = available.index.tolist()

                def match_label(index):
                    row = predictions.loc[index]
                    return f"{row['HomeTeam']}  vs  {row['AwayTeam']}"

                chosen_index = st.selectbox(
                    "Fixture",
                    fixture_indices,
                    format_func=match_label,
                )
                st.divider()
                st.caption("Estimates are generated from historical match data.")

            match = predictions.loc[chosen_index]
            match_date = match["Date"].strftime("%A, %d %B %Y")
            home_team = safe_text(match["HomeTeam"])
            away_team = safe_text(match["AwayTeam"])
            home_badge = safe_text(initials(match["HomeTeam"]))
            away_badge = safe_text(initials(match["AwayTeam"]))
            score = safe_text(match.get("PredictedScore", "—"))
            result = safe_text(match.get("PredictedResult", "—"))
            result_name = {"H": "Home win", "D": "Draw", "A": "Away win"}.get(result, result)

            st.markdown(
                f"""
                <div class="match-card">
                  <div class="match-meta"><span>{safe_text(match_date)}</span><span class="league-pill">ENGLAND · PREMIER LEAGUE</span></div>
                  <div class="teams">
                    <div class="team"><div class="team-badge">{home_badge}</div><div class="team-name">{home_team}</div></div>
                    <div class="vs">VS</div>
                    <div class="team away"><div class="team-name">{away_team}</div><div class="team-badge">{away_badge}</div></div>
                  </div>
                  <div class="prediction-line"><span>MODEL SCORE</span><span class="score-pill">{score}</span><span class="pred-tag">{result_name}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-title">Match result probabilities</div>', unsafe_allow_html=True)
            result_columns = st.columns(3)
            for col, label, color, probability_column in zip(
                result_columns,
                ["Home win", "Draw", "Away win"],
                ["var(--pink)", "var(--mint)", "var(--blue)"],
                ["Probability_H", "Probability_D", "Probability_A"],
            ):
                with col:
                    render_probability_card(label, match.get(probability_column), color)

            st.markdown('<div class="section-title">Expected corners</div>', unsafe_allow_html=True)
            corner_columns = st.columns(2)
            for col, label, key in zip(
                corner_columns,
                [f"{match['HomeTeam']} corners", f"{match['AwayTeam']} corners"],
                ["PredictedHomeCorners", "PredictedAwayCorners"],
            ):
                value = match.get(key)
                with col:
                    st.metric(label, "—" if value is None or pd.isna(value) else f"{float(value):.1f}")

            st.markdown('<div class="section-title">Goal markets</div>', unsafe_allow_html=True)
            market_items = [
                ("Double chance · Home or draw", "Probability_1X", None),
                ("Double chance · Draw or away", "Probability_X2", None),
                ("Double chance · Either team wins", "Probability_12", None),
                ("Both teams to score", "Probability_BTTS", "Prediction_BTTS"),
                ("BTTS + over 2.5 goals", "Probability_BTTS_Over2_5", "Prediction_BTTS_Over2_5"),
                ("Home win + over 1.5 goals", "Probability_HomeWin_Over1_5", "Prediction_HomeWin_Over1_5"),
            ]
            market_items = [item for item in market_items if item[1] in predictions.columns]
            for start in range(0, len(market_items), 3):
                market_row = st.columns(3)
                for col, (label, probability_column, prediction_column) in zip(
                    market_row, market_items[start:start + 3]
                ):
                    with col:
                        render_market_card(
                            label,
                            match.get(probability_column),
                            match.get(prediction_column) if prediction_column else None,
                        )

            st.markdown('<div class="section-title">All scheduled predictions</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-note">Select a fixture in the left panel to see its full prediction details.</div>', unsafe_allow_html=True)
            for _, row in predictions.iterrows():
                date_text = row["Date"].strftime("%d %b")
                score_text = safe_text(row.get("PredictedScore", "—"))
                st.markdown(
                    f"""
                    <div class="schedule-card">
                      <div class="schedule-date">{safe_text(date_text)}</div>
                      <div class="schedule-team">{safe_text(row['HomeTeam'])}</div>
                      <div class="schedule-vs">VS</div>
                      <div class="schedule-team right">{safe_text(row['AwayTeam'])}</div>
                      <div class="schedule-score">{score_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="footnote">Predictions are statistical estimates based on historical data. They are not guaranteed match outcomes.</div>',
                unsafe_allow_html=True,
            )

except Exception as error:
    st.error("The model could not load its predictions.")
    st.code(str(error))
    st.write(
        "Check that app.py, football_prediction.py, and the data folder "
        "are in the same project folder, then run the app with: "
        "python -m streamlit run app.py"
    )
