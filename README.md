# MatchEdge — Soccer Prediction Dashboard

An interactive Streamlit app that retrieves fixtures, recent results and fixture-level statistics from API-FOOTBALL (API-Sports), and estimates outcomes, bookings and corners.

## Install on Windows

1. Install Python 3.11 or 3.12.
2. Open Command Prompt in this folder.
3. Run `py -m pip install -r requirements.txt`.
4. Create `.streamlit/secrets.toml` and enter `API_FOOTBALL_KEY = "YOUR_KEY"` (or enter the key in the sidebar at runtime).
5. Run `py -m streamlit run app.py`.

Get an API key at https://www.api-football.com/ . API coverage, request quotas, and season availability vary by subscription. API-FOOTBALL uses https://v3.football.api-sports.io .

## How it works

- Fetches scheduled fixtures for a date, league and season.
- Fetches up to N recent completed fixtures for each team and each fixture's match statistics, where available.
- Uses shrinkage-adjusted recent home/away scoring rates and independent Poisson goals to estimate 1X2 probabilities and scorelines.
- Uses recent team-level corners and yellow-card totals to estimate expected match counts. Models count totals as Poisson for transparent over/under probabilities; cards/corners may be overdispersed, so estimates are indicative.
- Displays data completeness and does not invent cards/corners where source statistics are missing.

## Limitations

- Historical data access and fixtures depend on the API plan; each match statistics request counts toward quota.
- Recent fixtures may include other competitions; optional historical form is not a league-only model.
- The model is an interpretable baseline, not a trained or backtested betting model. Fair odds are inverse model probabilities with no bookmaker margin.
- API data can be delayed or corrected. Yellow cards are match-statistic counts, not disciplinary booking points.
- The app does not place bets, guarantee profits, or automatically update when closed.


## Current app
The hosted app provides fixture predictions only. The historical sample selector, historical backtesting tab, and model comparison interface have been removed. Prediction tools load only after a request. API coverage depends on the configured subscription.
