"""Transparent baseline count models for soccer predictions."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import poisson


def shrunk_mean(values, prior, strength=5):
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return (sum(clean) + strength * prior) / (len(clean) + strength)


def result_prediction(home_history, away_history):
    """History records have gf, ga, and home boolean."""
    hh = [x for x in home_history if x['home']]
    aa = [x for x in away_history if not x['home']]
    if len(hh) < 3: hh = home_history
    if len(aa) < 3: aa = away_history
    # Generic baselines are model priors, not observed league averages.
    h_for = shrunk_mean([x['gf'] for x in hh], 1.45)
    h_against = shrunk_mean([x['ga'] for x in hh], 1.15)
    a_for = shrunk_mean([x['gf'] for x in aa], 1.15)
    a_against = shrunk_mean([x['ga'] for x in aa], 1.45)
    lam_h = float(np.clip((h_for + a_against) / 2, 0.2, 4.5))
    lam_a = float(np.clip((a_for + h_against) / 2, 0.2, 4.5))
    # 0..12 grid captures almost all probability at the capped lambdas.
    g = np.arange(13)
    grid = np.outer(poisson.pmf(g, lam_h), poisson.pmf(g, lam_a))
    grid /= grid.sum()
    p_home = float(np.tril(grid, -1).sum())
    p_draw = float(np.trace(grid))
    p_away = float(np.triu(grid, 1).sum())
    idx = np.unravel_index(np.argmax(grid), grid.shape)
    return {'home_goals': lam_h, 'away_goals': lam_a, 'home_win': p_home,
            'draw': p_draw, 'away_win': p_away, 'score': f'{idx[0]}–{idx[1]}',
            'history_home': len(home_history), 'history_away': len(away_history)}


def count_prediction(home_history, away_history, field, prior_team, thresholds):
    h_for = shrunk_mean([x.get(field + '_for') for x in home_history], prior_team)
    a_for = shrunk_mean([x.get(field + '_for') for x in away_history], prior_team)
    h_against = shrunk_mean([x.get(field + '_against') for x in home_history], prior_team)
    a_against = shrunk_mean([x.get(field + '_against') for x in away_history], prior_team)
    h = (h_for + a_against) / 2
    a = (a_for + h_against) / 2
    total = h + a
    # Over n.5 means X >= n+1.
    overs = {f'Over {t:.1f}': float(poisson.sf(math.floor(t), total)) for t in thresholds}
    n_h = sum(x.get(field + '_for') is not None for x in home_history)
    n_a = sum(x.get(field + '_for') is not None for x in away_history)
    return {'home': h, 'away': a, 'total': total, 'overs': overs,
            'samples_home': n_h, 'samples_away': n_a,
            'data_available': n_h > 0 and n_a > 0}
