"""API-FOOTBALL v3 client with Streamlit caching."""
import requests
import streamlit as st
BASE = 'https://v3.football.api-sports.io'

class APIError(Exception): pass

@st.cache_data(ttl=900, show_spinner=False)
def fetch(endpoint, params, key):
    if endpoint not in {'/fixtures', '/fixtures/statistics', '/status'}:
        raise APIError('Unsupported API endpoint.')
    if not isinstance(key, str) or not key.strip():
        raise APIError('Configure an API-FOOTBALL key before requesting data.')
    try:
        response = requests.get(BASE + endpoint, params=dict(params),
                                headers={'x-apisports-key': key}, timeout=25, allow_redirects=False)
        if response.status_code != 200:
            raise APIError(f'API request failed (HTTP {response.status_code}). Check your key, plan and quota.')
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise APIError('Unable to contact API-FOOTBALL or decode its response.') from None
    if not isinstance(payload, dict):
        raise APIError('Unexpected API-FOOTBALL response format.')
    if payload.get('errors'):
        raise APIError('API-FOOTBALL rejected this request. Check your subscription coverage and remaining quota.')
    records = payload.get('response')
    if not isinstance(records, list):
        raise APIError('Unexpected API-FOOTBALL data format.')
    paging = payload.get('paging', {})
    if isinstance(paging, dict) and paging.get('total', 1) > 1:
        raise APIError('Paginated data requires explicit retrieval; refusing an incomplete sample.')
    return records

def fixtures(key, league, season, date):
    return fetch('/fixtures', (('league', league), ('season', season), ('date', date)), key)

def recent(key, team_id, season, n, before=None):
    # For historical fixture dates, request the season and filter strictly before kickoff.
    # API `last` would otherwise include matches played AFTER the target fixture.
    params = (('team', team_id), ('season', season), ('status', 'FT'))
    if before is None:
        params += (('last', n),)
    records = fetch('/fixtures', params, key)
    records = [x for x in records if x.get('goals', {}).get('home') is not None
               and x.get('goals', {}).get('away') is not None
               and (before is None or x['fixture']['date'] < before)]
    return sorted(records, key=lambda x: x['fixture']['date'])[-n:]

def statistics(key, fixture_id):
    return fetch('/fixtures/statistics', (('fixture', fixture_id),), key)

def parse_stat(stats, name):
    for item in stats or []:
        if item.get('type') == name:
            val = item.get('value')
            try: return float(val) if val is not None else None
            except (TypeError, ValueError): return None
    return None

def history(key, team_id, season, n, before=None):
    rows = []
    for match in recent(key, team_id, season, n, before=before):
        home_id = match['teams']['home']['id']
        is_home = home_id == team_id
        opp_id = match['teams']['away']['id'] if is_home else home_id
        gh, ga = match['goals']['home'], match['goals']['away']
        row = {'home': is_home, 'gf': gh if is_home else ga,
               'ga': ga if is_home else gh, 'date': match['fixture']['date'],
               'opponent': match['teams']['away']['name'] if is_home else match['teams']['home']['name']}
        stats = statistics(key, match['fixture']['id'])
        by_id = {x['team']['id']: x.get('statistics', []) for x in stats if x.get('team')}
        own, opp = by_id.get(team_id, []), by_id.get(opp_id, [])
        for field, api_field in [('corners', 'Corner Kicks'), ('cards', 'Yellow Cards')]:
            row[field + '_for'] = parse_stat(own, api_field)
            row[field + '_against'] = parse_stat(opp, api_field)
        rows.append(row)
    return rows
