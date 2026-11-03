from datetime import date, datetime
import os
import logging
from time import perf_counter

started = perf_counter()
logger = logging.getLogger("matchedge.startup")
logger.setLevel(logging.INFO)
logger.info("MatchEdge: app script started")
import streamlit as st

st.set_page_config(page_title='MatchEdge | Soccer Predictions', page_icon='⚽', layout='wide')
st.markdown('''<style>.block-container{padding-top:1.4rem}.stMetric{background:#12263a0c;border-radius:12px;padding:12px}</style>''', unsafe_allow_html=True)
st.title('⚽ MatchEdge')
st.caption('Live fixture lookup · historical form · probabilistic match, yellow-card and corner forecasts')
st.success('Ready — data and prediction tools load only when you request them.')
logger.info("MatchEdge: page shell rendered in %.3fs", perf_counter() - started)
if st.query_params.get('diagnostics') == '1':
    st.write('Page startup check passed. No API calls or prediction libraries were loaded.')
    st.caption(f'Page startup: {perf_counter() - started:.3f} seconds')
    st.stop()

def load_tools():
    tool_started = perf_counter()
    logger.info("MatchEdge: loading prediction tools")
    try:
        import pandas as pd
        import plotly.graph_objects as go
        from api import APIError, fixtures, history
        from model import result_prediction, count_prediction
        from backtest import completed_fixtures, walk_forward, evaluate
    except Exception:
        logger.exception("MatchEdge: prediction tools failed to load")
        st.error('The page is ready, but prediction tools could not load. Please check the app logs.')
        st.stop()
    logger.info("MatchEdge: prediction tools loaded in %.3fs", perf_counter() - tool_started)
    return pd, go, APIError, fixtures, history, result_prediction, count_prediction, completed_fixtures, walk_forward, evaluate

LEAGUES = {'UEFA Champions League':2, 'Europa League':3, 'Premier League':39,
           'La Liga':140, 'Serie A':135, 'Bundesliga':78, 'Ligue 1':61,
           'Saudi Pro League':307, 'CAF Champions League':12}
try: secret_key = st.secrets.get('API_FOOTBALL_KEY', '')
except Exception: secret_key = ''
with st.sidebar:
    st.header('Data source')
    key = os.getenv('API_FOOTBALL_KEY', secret_key).strip()
    if key:
        st.caption('API-FOOTBALL is configured securely on the server.')
    else:
        key = st.text_input('API-FOOTBALL key', type='password', help='Your own API-Sports v3 key; held only for this session.').strip()
    league_name = st.selectbox('Competition', list(LEAGUES))
    data_mode = st.selectbox('Data access', ['Historical sample (free plan)', 'Current dates (requires season access)'])
    default_date = date(2024, 11, 5) if data_mode.startswith('Historical') else date.today()
    date_text = st.text_input('Fixture date (YYYY-MM-DD)', value=default_date.isoformat(), key='fixture_date_' + data_mode)
    try:
        selected_date = date.fromisoformat(date_text.strip())
    except ValueError:
        st.error('Enter a valid fixture date in YYYY-MM-DD format, for example 2024-11-05.')
        st.stop()
    if data_mode.startswith('Historical'):
        st.caption('Historical sample: 5 November 2024. Free API plans restrict available seasons; current fixtures may need a paid plan.')
    season = st.number_input('Season start year', min_value=2018, max_value=2035, value=selected_date.year if selected_date.month >= 7 else selected_date.year - 1, step=1)
    n = st.number_input('Recent completed matches per team', min_value=3, max_value=15, value=3, step=1)
    max_games = st.number_input('Maximum fixtures to analyse', min_value=1, max_value=12, value=1, step=1)
    st.caption('Each team history loads match-level statistics. API usage can be substantial.')
    run = st.button('🔄 Fetch fixtures & predict', type='primary', use_container_width=True)
    st.divider()
    st.info('No API key? Obtain one at api-football.com. Live and historical coverage depend on your plan.')

tab_forecast, tab_backtest = st.tabs(['🔮 Predictions', '📊 Historical backtesting'])
with tab_backtest:
    st.subheader('Walk-forward historical backtest')
    st.write('Each match is predicted using only earlier completed fixtures in the selected competition. Games at identical kickoff times cannot train one another. Results are evaluated against final scores and observed yellow cards/corners.')
    bt_season = st.number_input('Backtest season start year', min_value=2018, max_value=2035, value=int(season), key='bt_season')
    bt_window = st.number_input('Prior games per team', min_value=3, max_value=20, value=8, step=1, key='bt_window')
    bt_min = st.number_input('Minimum prior games per team', min_value=1, max_value=10, value=3, step=1, key='bt_min')
    bt_limit = st.number_input('Maximum recent games to score', min_value=20, max_value=300, value=80, step=20, key='bt_limit')
    ml_compare = st.checkbox('Include Random Forest and XGBoost comparison', value=False, help='Loads additional tools only when selected. Baseline backtesting runs without them.')
    ml_min = st.number_input('ML minimum earlier training fixtures', min_value=20, max_value=150, value=40, step=10)
    ml_refit = st.number_input('ML refit interval (kickoff groups)', min_value=1, max_value=30, value=10, step=1)
    st.warning('API usage: this backtest may request statistics for EVERY completed fixture in the season, not just the scored fixtures. Check your provider quota before running. Results are cached for 15 minutes.')
    if st.button('Run historical backtest', type='primary', key='bt_run'):
        if not key:
            st.error('Enter your API-FOOTBALL key in the sidebar.')
        else:
            with st.spinner('Loading prediction tools…'):
                pd, go, APIError, fixtures, history, result_prediction, count_prediction, completed_fixtures, walk_forward, evaluate = load_tools()
            try:
                with st.spinner('Loading season fixtures…'):
                    past = completed_fixtures(key, LEAGUES[league_name], int(bt_season))
                if not past:
                    st.warning('No completed fixtures available for this season and league.')
                else:
                    prog = st.progress(0, text='Evaluating past matches…')
                    back = walk_forward(past, key, bt_window, bt_min, bt_limit,
                                        lambda i, total: prog.progress(i / total, text=f'Processed {i}/{total} historical matches'))
                    prog.empty()
                    scores, confusion = evaluate(back)
                    if back.empty:
                        st.warning('Not enough prior matches to evaluate. Lower the minimum history or select another season.')
                    else:
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric('Evaluated matches', scores['Matches evaluated'])
                        m2.metric('Result accuracy', f"{scores['Result accuracy']:.1%}")
                        m3.metric('Corners MAE', f"{scores['Corners MAE']:.2f}" if pd.notna(scores['Corners MAE']) else 'N/A')
                        m4.metric('Cards MAE', f"{scores['Cards MAE']:.2f}" if pd.notna(scores['Cards MAE']) else 'N/A')
                        st.markdown('#### All validation metrics')
                        st.dataframe(pd.DataFrame([{'Metric': k, 'Value': v} for k, v in scores.items()]), hide_index=True, use_container_width=True)
                        st.markdown('#### Result confusion matrix')
                        st.dataframe(confusion, use_container_width=True)
                        st.markdown('#### Actual vs predicted counts')
                        for label in ['corners', 'cards']:
                            pair = back[[f'Actual {label}', f'Expected {label}']].dropna()
                            if len(pair):
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=pair.iloc[:,1], y=pair.iloc[:,0], mode='markers', name=label))
                                fig.add_trace(go.Scatter(x=[0, max(pair.max())], y=[0, max(pair.max())], mode='lines', name='Perfect prediction'))
                                fig.update_layout(title=label.title(), xaxis_title='Predicted total', yaxis_title='Actual total', height=320)
                                st.plotly_chart(fig, use_container_width=True)
                        st.markdown('#### Match-by-match backtest')
                        st.dataframe(back, hide_index=True, use_container_width=True)
                        st.download_button('Download backtest CSV', back.to_csv(index=False).encode('utf-8'), file_name=f'matchedge_backtest_{league_name.replace(" ", "_")}_{bt_season}.csv', mime='text/csv')
                        st.caption('Brier/log loss: lower is better. MAE/RMSE: lower is better. Bias = prediction minus actual. Missing card/corner observations are excluded, not imputed. These are retrospective out-of-sample predictions, not a guarantee of future performance.')
                        if ml_compare:
                            st.markdown('### Model comparison: Poisson vs Random Forest vs XGBoost')
                            with st.spinner('Loading and training comparison models…'):
                                from compare import compare_models
                                comparison_rows, comparison_metrics = compare_models(back, min_train=ml_min, refit_every=ml_refit)
                            if comparison_metrics.empty:
                                st.info('Not enough earlier matches to train and test the ML models. Increase the backtest sample or lower minimum training fixtures.')
                            else:
                                st.dataframe(comparison_metrics, use_container_width=True, hide_index=True)
                                for metric in ['Result accuracy', 'Multiclass Brier score', 'Corners MAE', 'Cards MAE']:
                                    chart = comparison_metrics[['Model',metric]].dropna()
                                    if not chart.empty:
                                        st.bar_chart(chart.set_index('Model'), y=metric)
                                st.download_button('Download comparative predictions CSV', comparison_rows.to_csv(index=False).encode('utf-8'), file_name='matchedge_model_comparison_predictions.csv', mime='text/csv')
                                st.download_button('Download model comparison metrics CSV', comparison_metrics.to_csv(index=False).encode('utf-8'), file_name='matchedge_model_comparison_metrics.csv', mime='text/csv')
                                st.caption('All models are compared on the same chronologically eligible fixtures. ML models are refitted using only older kickoff groups. ML cards/corners outputs are point forecasts; over/under probability scores are not reported for those regressors. No hyperparameter tuning on test data.')

            except APIError as exc:
                st.error(str(exc))
with tab_forecast:
    if not run:
        st.markdown('### Ready to analyse')
        st.write('Choose a competition and fixture date, enter your API key, then click **Fetch fixtures & predict**.')
    elif not key:
        st.error('Enter an API-FOOTBALL key in the sidebar to fetch real data.')
    else:
        with st.spinner('Loading prediction tools…'):
            pd, go, APIError, fixtures, history, result_prediction, count_prediction, completed_fixtures, walk_forward, evaluate = load_tools()
        try:
            with st.spinner('Fetching scheduled fixtures…'):
                games = fixtures(key, LEAGUES[league_name], int(season), selected_date.isoformat())
        except APIError as exc:
            st.error(str(exc)); st.stop()
        if not games:
            st.warning('No fixtures returned for this competition/date/season. Check the season or your API coverage.')
            st.stop()

        games = [g for g in games if g.get('teams', {}).get('home') and g.get('teams', {}).get('away')]
        if not games:
            st.warning('No usable fixtures returned.'); st.stop()
        if len(games) > max_games:
            st.info(f'{len(games)} fixtures returned; analysing the first {max_games}. Increase the sidebar limit to include more.')

        rows = []
        details = {}
        cache_hist = {}
        progress = st.progress(0, text='Analysing fixtures…')
        for i, g in enumerate(games[:max_games]):
            home = g['teams']['home']; away = g['teams']['away']
            try:
                for team in (home, away):
                    if (team['id'], g['fixture']['date']) not in cache_hist:
                        cache_hist[(team['id'], g['fixture']['date'])] = history(key, team['id'], int(season), n, before=g['fixture']['date'])
                hh, aa = cache_hist[(home['id'], g['fixture']['date'])], cache_hist[(away['id'], g['fixture']['date'])]
                result = result_prediction(hh, aa)
                corners = count_prediction(hh, aa, 'corners', 4.7, [7.5,8.5,9.5,10.5])
                cards = count_prediction(hh, aa, 'cards', 2.0, [2.5,3.5,4.5,5.5])
                probs = {'Home': result['home_win'], 'Draw': result['draw'], 'Away': result['away_win']}
                pick = max(probs, key=probs.get)
                kickoff = pd.to_datetime(g['fixture']['date'], utc=True).strftime('%Y-%m-%d %H:%M UTC')
                name = f"{home['name']} vs {away['name']}"
                row = {'Kickoff (UTC)':kickoff, 'Match':name, 'Pick':pick, 'Predicted score':result['score'],
                       'Home win %':round(result['home_win']*100,1), 'Draw %':round(result['draw']*100,1),
                       'Away win %':round(result['away_win']*100,1),
                       'Fair home odds':round(1/result['home_win'],2), 'Fair draw odds':round(1/result['draw'],2),
                       'Fair away odds':round(1/result['away_win'],2),
                       'Expected home goals':round(result['home_goals'],2), 'Expected away goals':round(result['away_goals'],2),
                       'Expected corners':round(corners['total'],2) if corners['data_available'] else None,
                       'Expected yellow cards':round(cards['total'],2) if cards['data_available'] else None,
                       'Corner sample (H/A)':f"{corners['samples_home']}/{corners['samples_away']}",
                       'Card sample (H/A)':f"{cards['samples_home']}/{cards['samples_away']}",
                       'Form sample (H/A)':f"{len(hh)}/{len(aa)}"}
                rows.append(row)
                details[name] = (result,corners,cards,hh,aa,home['name'],away['name'])
            except (APIError, KeyError, ValueError, TypeError) as exc:
                st.warning(f"Skipped {home['name']} vs {away['name']}: {exc}")
            progress.progress((i+1)/min(len(games),max_games), text=f'Analysed {i+1} fixture(s)')
        progress.empty()
        if not rows:
            st.error('No predictions could be calculated. Check your API quota and season access.'); st.stop()

        df = pd.DataFrame(rows)
        a,b,c = st.columns(3)
        a.metric('Fixtures analysed',len(df))
        b.metric('With corners data',df['Expected corners'].notna().sum())
        c.metric('With cards data',df['Expected yellow cards'].notna().sum())
        st.subheader('Prediction table')
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button('⬇ Download predictions CSV',df.to_csv(index=False).encode('utf-8'),file_name=f'matchedge_{selected_date}.csv',mime='text/csv')

        st.subheader('Match detail')
        selected = st.selectbox('Select match',list(details))
        result,corners,cards,hh,aa,hn,an = details[selected]
        left,right = st.columns([1,1])
        with left:
            st.markdown('#### Match outcome probabilities')
            fig = go.Figure(go.Bar(x=['Home win','Draw','Away win'],y=[100*result['home_win'],100*result['draw'],100*result['away_win']],text=[f'{100*x:.1f}%' for x in [result['home_win'],result['draw'],result['away_win']]],textposition='auto'))
            fig.update_layout(yaxis_title='Probability (%)',yaxis_range=[0,100],height=330,margin=dict(l=10,r=10,t=15,b=10))
            st.plotly_chart(fig,use_container_width=True)
            st.caption(f"Expected goals: {hn} {result['home_goals']:.2f}, {an} {result['away_goals']:.2f}. Most likely score: {result['score']}.")
        with right:
            st.markdown('#### Corners & yellow cards')
            if corners['data_available']:
                st.metric('Expected total corners',f"{corners['total']:.1f}")
                st.write(f"{hn}: {corners['home']:.1f} · {an}: {corners['away']:.1f}")
                st.dataframe(pd.DataFrame({'Market':list(corners['overs']), 'Model probability %':[round(100*v,1) for v in corners['overs'].values()]}),hide_index=True,use_container_width=True)
            else: st.warning('Insufficient corner statistics for both teams. No corner forecast displayed.')
            if cards['data_available']:
                st.metric('Expected yellow cards',f"{cards['total']:.1f}")
                st.write(f"{hn}: {cards['home']:.1f} · {an}: {cards['away']:.1f}")
                st.dataframe(pd.DataFrame({'Market':list(cards['overs']), 'Model probability %':[round(100*v,1) for v in cards['overs'].values()]}),hide_index=True,use_container_width=True)
            else: st.warning('Insufficient yellow-card statistics for both teams. No card forecast displayed.')

        with st.expander('Historical form and data quality'):
            for team,records in [(hn,hh),(an,aa)]:
                st.markdown(f'**{team} — {len(records)} completed fixtures**')
                if records:
                    hdf = pd.DataFrame(records)
                    st.dataframe(hdf[['date','opponent','home','gf','ga','corners_for','corners_against','cards_for','cards_against']],use_container_width=True,hide_index=True)
                else: st.caption('No recent results returned for this team.')
            st.caption('Historical fixtures can include other competitions. Counts are raw fixture statistics; API completeness varies.')

        st.divider()
        st.caption('Model methodology: recent form shrunk toward generic priors; independent Poisson goals; Poisson card/corner totals. Not backtested or calibrated. Fair odds exclude margin. For research/decision support, not a guarantee of betting outcomes.')

logger.info("MatchEdge: app render finished in %.3fs", perf_counter() - started)
