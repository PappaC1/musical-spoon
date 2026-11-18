import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.stats import poisson
from datetime import datetime, timedelta
import difflib
import re

st.set_page_config(page_title="Soccer Challenge Hub", layout="centered")

API_KEY = st.secrets["API_FOOTBALL_KEY"]
HEADERS = {"x-apisports-key": API_KEY}
BASE_URL = "https://api-sports.io"


@st.cache_data(ttl=3600)
def get_fixtures_for_date(date_str):
    url = f"{BASE_URL}/fixtures"
    params = {"date": date_str, "timezone": "Africa/Johannesburg"}
    response = requests.get(url, headers=HEADERS, params=params)
    try:
        data = response.json()
        if data.get("errors"):
            st.error(f"API Message: {data['errors']}")
        return data.get("response", [])
    except Exception:
        st.error(f"API Connection Error (Status {response.status_code}): Check BASE_URL.")
        return []

def get_two_day_fixtures(start_date):
    day_one = start_date.strftime("%Y-%m-%d")
    day_two = (start_date + timedelta(days=1)).strftime("%Y-%m-%d")
    return get_fixtures_for_date(day_one) + get_fixtures_for_date(day_two)

@st.cache_data(ttl=86400)
def get_team_goal_rates(team_id, is_home):
    url = f"{BASE_URL}/fixtures"
    params = {"team": team_id, "last": 10, "status": "FT"}
    res = requests.get(url, headers=HEADERS, params=params).json()
    matches = res.get("response", [])

    scored, conceded = [], []
    for m in matches:
        home_match = m["teams"]["home"]["id"] == team_id
        if is_home and not home_match:
            continue
        if not is_home and home_match:
            continue

        gh = m["goals"]["home"] or 0
        ga = m["goals"]["away"] or 0
        if home_match:
            scored.append(gh)
            conceded.append(ga)
        else:
            scored.append(ga)
            conceded.append(gh)

    avg_scored = np.mean(scored) if scored else 1.2
    avg_conceded = np.mean(conceded) if conceded else 1.2
    return float(avg_scored), float(avg_conceded)

@st.cache_data(ttl=3600)
def get_match_odds(fixture_id):
    url = f"{BASE_URL}/odds"
    params = {"fixture": fixture_id}
    res = requests.get(url, headers=HEADERS, params=params).json()
    try:
        bookmakers = res["response"]["bookmakers"]
        for bm in bookmakers:
            for bet in bm["bets"]:
                if bet["name"] == "Match Winner":
                    return {item["value"]: float(item["odd"]) for item in bet["values"]}
    except (IndexError, KeyError, TypeError, ValueError):
        pass
    return {}

def predict_margins(home_xg, away_xg, max_goals=6):
    home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals)]
    away_probs = [poisson.pmf(j, away_xg) for j in range(max_goals)]
    matrix = np.outer(home_probs, away_probs)

    margins = {
        "Home by 3+": 0.0,
        "Home by 2": 0.0,
        "Home by 1": 0.0,
        "Draw": 0.0,
        "Away by 1": 0.0,
        "Away by 2": 0.0,
        "Away by 3+": 0.0,
    }
    outcome_probs = {"Home": 0.0, "Draw": 0.0, "Away": 0.0}
    scorelines = []

    for h in range(max_goals):
        for a in range(max_goals):
            p = matrix[h, a]
            scorelines.append((f"{h}-{a}", p))
            diff = h - a
            if diff > 0:
                outcome_probs["Home"] += p
            elif diff == 0:
                outcome_probs["Draw"] += p
            else:
                outcome_probs["Away"] += p

            if diff >= 3:
                margins["Home by 3+"] += p
            elif diff == 2:
                margins["Home by 2"] += p
            elif diff == 1:
                margins["Home by 1"] += p
            elif diff == 0:
                margins["Draw"] += p
            elif diff == -1:
                margins["Away by 1"] += p
            elif diff == -2:
                margins["Away by 2"] += p
            else:
                margins["Away by 3+"] += p

    top_scores = sorted(scorelines, key=lambda x: x, reverse=True)[:3]
    return margins, top_scores, outcome_probs

def evaluate_value(outcome_probs, odds, threshold=0.05):
    if not odds:
        return "No Odds", []

    alerts = []
    best_edge = -1.0
    best_label = "-"

    for label, code in [("Home", "1"), ("Draw", "X"), ("Away", "2")]:
        odd = odds.get(label)
        if odd and odd > 1.0:
            implied_prob = 1.0 / odd
            model_prob = outcome_probs.get(label, 0.0)
            edge = model_prob - implied_prob
            if edge >= threshold:
                alerts.append(f"🔥 {label} ({code}) @ {odd:.2f} (+{edge*100:.0f}% edge)")
                if edge > best_edge:
                    best_edge = edge
                    best_label = f"🔥 {code} @{odd:.2f}"

    return best_label, alerts

def calculate_match_data(fixture_id, home_id, away_id):
    h_att, h_def = get_team_goal_rates(home_id, is_home=True)
    a_att, a_def = get_team_goal_rates(away_id, is_home=False)

    home_xg = (h_att + a_def) / 2
    away_xg = (a_att + h_def) / 2

    margins, top_scores, outcome_probs = predict_margins(home_xg, away_xg)
    best_margin = max(margins.items(), key=lambda x: x)
    odds = get_match_odds(fixture_id)
    value_tag, value_alerts = evaluate_value(outcome_probs, odds)

    return {
        "home_xg": home_xg,
        "away_xg": away_xg,
        "margins": margins,
        "top_scores": top_scores,
        "outcome_probs": outcome_probs,
        "best_margin": best_margin,
        "odds": odds,
        "value_tag": value_tag,
        "value_alerts": value_alerts
    }

def filter_challenge_round(fixtures, raw_paste):
    if not raw_paste.strip():
        return fixtures

    lines = [line.strip() for line in raw_paste.splitlines() if len(line.strip()) > 2]
    candidates = []
    for line in lines:
        parts = re.split(r"\s+vs\.?\s+|\s+v\s+|\s+-\s+", line, flags=re.IGNORECASE)
        candidates.extend([p.strip() for p in parts if len(p.strip()) > 2])

    api_team_map = {}
    for m in fixtures:
        api_team_map[m["teams"]["home"]["name"].lower()] = m
        api_team_map[m["teams"]["away"]["name"].lower()] = m

    api_names = list(api_team_map.keys())
    matched_fixtures = []
    seen_ids = set()

    for cand in candidates:
        matches = difflib.get_close_matches(cand.lower(), api_names, n=1, cutoff=0.65)
        if matches:
            fix = api_team_map[matches]
            fid = fix["fixture"]["id"]
            if fid not in seen_ids:
                seen_ids.add(fid)
                matched_fixtures.append(fix)

    return matched_fixtures

def render_prediction_card(data):
    best_margin = data["best_margin"]

    if data["value_alerts"]:
        for alert in data["value_alerts"]:
            st.success(f"**VALUE ALERT:** {alert}")
    elif not data["odds"]:
        st.caption("Pre-match odds not published yet for this fixture.")

    c1, c2 = st.columns(2)
    c1.metric("Top Margin Pick", best_margin, f"{best_margin*100:.1f}%")
    c2.metric("Expected Goals (xG)", f"{data['home_xg']:.2f} - {data['away_xg']:.2f}")

    if data["odds"]:
        st.write("**1X2 Odds vs. Poisson Model:**")
        o_cols = st.columns(3)
        for idx, (label, short) in enumerate([("Home", "1"), ("Draw", "X"), ("Away", "2")]):
            odd = data["odds"].get(label, 0)
            m_prob = data["outcome_probs"][label] * 100
            o_cols[idx].metric(f"{short} ({odd:.2f})", f"{m_prob:.0f}% Model")

    st.write("**Top 3 Most Likely Scorelines:**")
    s_cols = st.columns(3)
    for s_idx, (score, prob) in enumerate(data["top_scores"]):
        s_cols[s_idx].metric(score, f"{prob*100:.1f}%")

    st.bar_chart(data["margins"])

st.title("⚽ Soccer Challenge Hub")

if "run_all" not in st.session_state:
    st.session_state.run_all = False

selected_date = st.date_input("Round Start Date (Loads 2-Day Window)", datetime.today())
fixtures = get_two_day_fixtures(selected_date)

with st.expander("📋 Filter Hollywood Soccer Challenge Round", expanded=True):
    pasted_card = st.text_area(
        "Paste your Hollywoodbets Soccer Challenge fixture list here:",
        height=120,
        placeholder="Paste raw text copied from Hollywoodbets Soccer Challenge..."
    )

active_fixtures = filter_challenge_round(fixtures, pasted_card)

if pasted_card.strip():
    st.success(f"Matched {len(active_fixtures)} Challenge matches across the 2-day window.")
    if active_fixtures:
        col_run, col_reset = st.columns(2)
        if col_run.button("⚡ Run All Challenge Matches", type="primary", use_container_width=True):
            st.session_state.run_all = True
        if col_reset.button("Reset View", use_container_width=True):
            st.session_state.run_all = False
else:
    st.session_state.run_all = False
    st.info(f"Loaded {len(fixtures)} global fixtures across 2 days. Paste your Challenge card above to filter.")

computed_results = {}
summary_rows = []

if st.session_state.run_all and active_fixtures:
    with st.spinner("Calculating Poisson margins & checking value odds..."):
        for idx, match in enumerate(active_fixtures, start=1):
            fid = match["fixture"]["id"]
            home = match["teams"]["home"]["name"]
            away = match["teams"]["away"]["name"]
            data = calculate_match_data(fid, match["teams"]["home"]["id"], match["teams"]["away"]["id"])
            computed_results[fid] = data

            summary_rows.append({
                "#": idx,
                "Match": f"{home} vs {away}",
                "Top Pick": f"{data['best_margin']} ({data['best_margin']*100:.0f}%)",
                "Value": data["value_tag"],
                "Score": f"{data['top_scores']} ({data['top_scores']*100:.0f}%)"
            })

    st.subheader("⚡ Quick Picks & Value Cheat Sheet")
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, hide_index=True, use_container_width=True)
    st.divider()

for idx, match in enumerate(active_fixtures, start=1):
    fid = match["fixture"]["id"]
    league = match["league"]["name"]
    country = match["league"]["country"]
    home = match["teams"]["home"]["name"]
    away = match["teams"]["away"]["name"]
    match_date = match["fixture"]["date"][:10]
    kickoff = match["fixture"]["date"][11:16]

    with st.container(border=True):
        st.caption(f"Match #{idx} • {country} ({league}) | 📅 {match_date} 🕒 {kickoff} SAST")
        st.subheader(f"{home} vs {away}")

        if fid in computed_results:
            render_prediction_card(computed_results[fid])
        else:
            if st.button("Run Poisson & Value Check", key=f"pred_{fid}"):
                data = calculate_match_data(fid, match["teams"]["home"]["id"], match["teams"]["away"]["id"])
                render_prediction_card(data)
