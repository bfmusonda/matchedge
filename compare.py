"""Chronological, expanding-window comparison of baseline, RF and XGBoost.

Consumes strictly pre-match features from walk_forward. No random train/test split.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from backtest import evaluate

FEATURES = ['Home GF form','Home GA form','Away GF form','Away GA form',
            'Home corners form','Home corners conceded','Away corners form','Away corners conceded',
            'Home cards form','Home cards conceded','Away cards form','Away cards conceded',
            'Home history','Away history']
LABELS = ['Home', 'Draw', 'Away']


def _model(kind, target, seed):
    if target == 'result':
        if kind == 'Random Forest':
            return RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=4,
                                          random_state=seed, n_jobs=-1)
        return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=.06,
                             subsample=.85, colsample_bytree=.85, random_state=seed,
                             n_jobs=2, objective='multi:softprob', eval_metric='mlogloss')
    if kind == 'Random Forest':
        return RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_leaf=4,
                                     random_state=seed, n_jobs=-1)
    return XGBRegressor(n_estimators=100, max_depth=3, learning_rate=.06,
                        subsample=.85, colsample_bytree=.85, random_state=seed, n_jobs=2,
                        objective='reg:squarederror')


def compare_models(back, min_train=40, refit_every=10, seed=42, progress=None):
    """Train on strictly earlier kickoff groups, score on later groups.

    All models are evaluated on identical eligible fixtures; counts use identical
    observed-target subsets. Refit happens only with older completed matches.
    """
    if back.empty:
        return pd.DataFrame(), pd.DataFrame()
    missing = [f for f in FEATURES if f not in back]
    if missing:
        raise ValueError('Backtest lacks prematch features: ' + ', '.join(missing))
    data = back.sort_values(['Date','Fixture']).reset_index(drop=True).copy()
    # Only rows with enough earlier completed matches can be evaluated.
    groups = list(data.groupby('Date', sort=True).indices.items())
    predictions = []
    cache = {}
    for gi, (stamp, idx) in enumerate(groups):
        idx = np.asarray(idx)
        train = data[data['Date'] < stamp]
        if len(train) < min_train:
            continue
        test = data.iloc[idx]
        # Refit at most every N kickoff groups; training set never contains
        # the prediction group's data (or anything later).
        if not cache or gi - cache['group'] >= refit_every:
            trained = {}
            x = train[FEATURES].astype(float).fillna(-1)
            y = train['Actual result'].map({v:i for i,v in enumerate(LABELS)})
            for kind in ('Random Forest','XGBoost'):
                for target in ('result','corners','cards'):
                    if target == 'result':
                        if len(set(y)) < 3:
                            continue
                        model = _model(kind, target, seed)
                        model.fit(x, y)
                    else:
                        label = 'Actual ' + target
                        ok = train[label].notna()
                        if ok.sum() < min_train:
                            continue
                        model = _model(kind, target, seed)
                        model.fit(x.loc[ok], train.loc[ok,label].astype(float))
                    trained[(kind,target)] = model
            cache = {'group':gi,'models':trained}
        xt = test[FEATURES].astype(float).fillna(-1)
        for kind in ('Poisson baseline','Random Forest','XGBoost'):
            out = test[['Date','Fixture','Actual result','Actual corners','Actual cards']].copy()
            out['Model'] = kind
            if kind == 'Poisson baseline':
                for k in LABELS:
                    out['P('+k+')'] = test['P('+k+')'].to_numpy()
                for k in ('corners','cards'):
                    out['Expected '+k] = test['Expected '+k].to_numpy()
                    threshold = 9.5 if k == 'corners' else 4.5
                    out[f'P({k} over {threshold})'] = test[f'P({k} over {threshold})'].to_numpy()
            else:
                m = cache['models'].get((kind,'result'))
                if m is None:
                    continue
                p = m.predict_proba(xt)
                for k in LABELS:
                    out['P('+k+')'] = p[:,list(m.classes_).index(LABELS.index(k))]
                for k in ('corners','cards'):
                    m = cache['models'].get((kind,k))
                    out['Expected '+k] = np.maximum(m.predict(xt),0) if m is not None else np.nan
                    # ML regressors are not probabilistic count models; do not
                    # manufacture over/under probabilities from point predictions.
                    threshold = 9.5 if k == 'corners' else 4.5
                    out[f'P({k} over {threshold})'] = np.nan
            out['Predicted result'] = [LABELS[i] for i in np.argmax(out[['P(Home)','P(Draw)','P(Away)']].to_numpy(),axis=1)]
            predictions.append(out)
        if progress:
            progress(gi+1,len(groups))
    if not predictions:
        return pd.DataFrame(), pd.DataFrame()
    scored = pd.concat(predictions,ignore_index=True)
    rows = []
    for name, part in scored.groupby('Model',sort=False):
        metrics,_ = evaluate(part)
        rows.append({'Model':name, **metrics})
    return scored, pd.DataFrame(rows)
