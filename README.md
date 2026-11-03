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

## Historical backtesting (new)
Open the **Historical backtesting** tab. Select a season, prior-games window, minimum prior history and evaluation limit, then run. The test scores the most recent selected fixtures but processes the full selected season chronologically to build earlier training histories. Matches sharing the same exact kickoff timestamp are predicted before any of their outcomes are added to history. This avoids look-ahead leakage. The backtest computes result accuracy, multiclass Brier score, log loss, confusion matrix, cards/corners MAE, RMSE, bias, and over-line Brier scores, plus per-match CSV export. Missing statistics are excluded from count metrics. No API key is bundled. API usage may be high because match statistics are requested for all completed fixtures in the selected season. The backtest is **within-season** only and has no prior-season warm-up; early fixtures with insufficient prior history are skipped. Models use fixed priors and are not fitted on the test set.

## Three-model chronological comparison

The Historical backtesting tab now compares the existing Poisson baseline with Random Forest and XGBoost for 1X2 outcomes, yellow-card totals, and corner totals. Each match's ML features are derived from prior team matches only. An expanding training window uses only earlier kickoff timestamps, with configurable minimum training size and refit interval. All three models are scored on identical eligible fixtures and observed count subsets. Metrics include result accuracy, multiclass Brier score/log loss, corners/cards MAE, RMSE, and bias; the baseline additionally reports over-line Brier scores. The regressors provide count estimates **not** calibrated over/under probabilities. Download both the per-match comparison and summary CSV. ML training begins after enough within-season eligible historical games are available; increase the maximum scored fixtures to obtain more test samples. This is an exploratory comparison with fixed model hyperparameters, not a tuned or validated betting system. Never use same-season test performance to select hyperparameters and report it as unbiased final performance; reserve a later season for final evaluation.

Note: This release uses the selected season only, without prior-season warm-up, and requires API match-statistics coverage. Predictions and historical accuracy cannot be populated without live API access. Local smoke testing used synthetic records and does not represent real-world model accuracy.

## Deployment and tests

See `DEPLOYMENT.md` for secure hosting instructions. Run `python -m pytest -q tests` after installing requirements. The app now excludes historical matches played at or after each target kickoff when computing form. This avoids hindsight in date-based forecasts. This release has not been validated with live provider credentials or deployed.
