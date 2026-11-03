"""Leakage-free chronological backtesting for MatchEdge baseline models."""
from collections import defaultdict
import numpy as np
import pandas as pd
from api import fetch, statistics, parse_stat
from model import result_prediction, count_prediction, shrunk_mean


def completed_fixtures(key, league, season):
    records = fetch('/fixtures', (('league', league), ('season', season), ('status', 'FT')), key)
    return sorted((g for g in records if g.get('goals', {}).get('home') is not None
                   and g.get('goals', {}).get('away') is not None),
                  key=lambda g: (g['fixture']['date'], g['fixture']['id']))


def build_records(game, key):
    """Return two historical rows and actual outcome; missing statistics remain None."""
    h, a = game['teams']['home'], game['teams']['away']
    gh, ga = int(game['goals']['home']), int(game['goals']['away'])
    stats = statistics(key, game['fixture']['id'])
    by_id = {s['team']['id']: s.get('statistics', []) for s in stats if s.get('team')}
    hs, ass = by_id.get(h['id'], []), by_id.get(a['id'], [])
    hc, ac = parse_stat(hs, 'Corner Kicks'), parse_stat(ass, 'Corner Kicks')
    hy, ay = parse_stat(hs, 'Yellow Cards'), parse_stat(ass, 'Yellow Cards')
    common = {'date': game['fixture']['date']}
    home = {**common, 'home': True, 'gf': gh, 'ga': ga, 'opponent': a['name'],
            'corners_for': hc, 'corners_against': ac, 'cards_for': hy, 'cards_against': ay}
    away = {**common, 'home': False, 'gf': ga, 'ga': gh, 'opponent': h['name'],
            'corners_for': ac, 'corners_against': hc, 'cards_for': ay, 'cards_against': hy}
    return home, away, hc, ac, hy, ay


def walk_forward(games, key, window=8, min_history=3, max_matches=120, progress=None):
    """Evaluate chronologically. Same kickoff timestamp is evaluated before updating history.

    A limited sample of the most recent max_matches fixtures is scored, but all
    supplied earlier fixtures can contribute training history. No future rows used.
    """
    games = sorted(games, key=lambda g: (g['fixture']['date'], g['fixture']['id']))
    eligible_ids = {g['fixture']['id'] for g in games[-max_matches:]}
    histories = defaultdict(list)
    output = []
    i = 0
    while i < len(games):
        stamp = games[i]['fixture']['date']
        group = []
        while i < len(games) and games[i]['fixture']['date'] == stamp:
            group.append(games[i]); i += 1
        pending = []
        for g in group:
            hid, aid = g['teams']['home']['id'], g['teams']['away']['id']
            hh, aa = histories[hid][-window:], histories[aid][-window:]
            eligible = g['fixture']['id'] in eligible_ids and len(hh) >= min_history and len(aa) >= min_history
            pred = None
            if eligible:
                r = result_prediction(hh, aa)
                c = count_prediction(hh, aa, 'corners', 4.7, [8.5, 9.5, 10.5])
                y = count_prediction(hh, aa, 'cards', 2.0, [3.5, 4.5, 5.5])
                pred = (r, c, y)
            pending.append((g, pred, hh, aa))
        # Actuals are loaded only AFTER all predictions for this kickoff were made.
        for g, pred, hh, aa in pending:
            home, away, hc, ac, hy, ay = build_records(g, key)
            if pred is not None:
                r, c, y = pred
                gh, ga = int(g['goals']['home']), int(g['goals']['away'])
                actual = 'Home' if gh > ga else 'Away' if ga > gh else 'Draw'
                probs = {'Home': r['home_win'], 'Draw': r['draw'], 'Away': r['away_win']}
                output.append({'Date': g['fixture']['date'], 'Fixture': f"{g['teams']['home']['name']} vs {g['teams']['away']['name']}",
                    'Actual result': actual, 'Predicted result': max(probs, key=probs.get),
                    'P(Home)': r['home_win'], 'P(Draw)': r['draw'], 'P(Away)': r['away_win'],
                    'Actual home goals': gh, 'Actual away goals': ga,
                    'Expected home goals': r['home_goals'], 'Expected away goals': r['away_goals'],
                    'Actual corners': hc + ac if hc is not None and ac is not None else np.nan,
                    'Expected corners': c['total'] if c['data_available'] else np.nan,
                    'Actual cards': hy + ay if hy is not None and ay is not None else np.nan,
                    'Expected cards': y['total'] if y['data_available'] else np.nan,
                    'P(corners over 9.5)': c['overs']['Over 9.5'] if c['data_available'] else np.nan,
                    'P(cards over 4.5)': y['overs']['Over 4.5'] if y['data_available'] else np.nan,
                    'Home GF form': shrunk_mean([v['gf'] for v in hh], 1.45),
                    'Home GA form': shrunk_mean([v['ga'] for v in hh], 1.15),
                    'Away GF form': shrunk_mean([v['gf'] for v in aa], 1.15),
                    'Away GA form': shrunk_mean([v['ga'] for v in aa], 1.45),
                    **{f'{prefix} {label}': shrunk_mean([v.get(field) for v in hist], prior)
                       for prefix,hist in [('Home',hh),('Away',aa)]
                       for label,field,prior in [('corners form','corners_for',4.7),('corners conceded','corners_against',4.7),
                                                 ('cards form','cards_for',2.0),('cards conceded','cards_against',2.0)]},
                    'Home history': len(histories[g['teams']['home']['id']]),
                    'Away history': len(histories[g['teams']['away']['id']])})
            histories[g['teams']['home']['id']].append(home)
            histories[g['teams']['away']['id']].append(away)
        if progress:
            progress(i, len(games))
    return pd.DataFrame(output)


def evaluate(df):
    if df.empty:
        return {}, pd.DataFrame()
    outcomes = ['Home', 'Draw', 'Away']
    p = df[['P(Home)', 'P(Draw)', 'P(Away)']].to_numpy(dtype=float)
    y = np.column_stack([(df['Actual result'] == k).to_numpy(dtype=float) for k in outcomes])
    metrics = {'Matches evaluated': len(df),
               'Result accuracy': float((df['Predicted result'] == df['Actual result']).mean()),
               'Multiclass Brier score': float(np.mean(np.sum((p-y)**2, axis=1))),
               'Result log loss': float(-np.mean(np.log(np.clip(p[np.arange(len(df)), y.argmax(axis=1)], 1e-12, 1))))}
    for label in ['corners', 'cards']:
        subset = df[[f'Actual {label}', f'Expected {label}']].dropna()
        metrics[f'{label.title()} sample'] = len(subset)
        metrics[f'{label.title()} MAE'] = float(np.abs(subset.iloc[:, 0]-subset.iloc[:, 1]).mean()) if len(subset) else np.nan
        metrics[f'{label.title()} RMSE'] = float(np.sqrt(np.mean((subset.iloc[:, 0]-subset.iloc[:, 1])**2))) if len(subset) else np.nan
        metrics[f'{label.title()} bias'] = float((subset.iloc[:, 1]-subset.iloc[:, 0]).mean()) if len(subset) else np.nan
        threshold = 9.5 if label == 'corners' else 4.5
        event = df[[f'Actual {label}', f'P({label} over {threshold})']].dropna()
        metrics[f'{label.title()} over {threshold} Brier'] = float(np.mean((event.iloc[:,1]-(event.iloc[:,0]>threshold).astype(float))**2)) if len(event) else np.nan
    confusion = pd.crosstab(pd.Categorical(df['Actual result'], categories=outcomes),
                            pd.Categorical(df['Predicted result'], categories=outcomes),
                            rownames=['Actual'], colnames=['Predicted'], dropna=False)
    return metrics, confusion
