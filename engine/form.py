"""
engine/form.py
==============
Berekent gewogen recente vorm per team op basis van:
- Historische WK-wedstrijden (2014/2018/2022)
- Gespeelde WK 2026 wedstrijden (tijdens toernooi)

Recentere wedstrijden wegen zwaarder via exponentiële decay.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Decay factor: hoe snel oudere wedstrijden minder wegen
# 0.85 = elke wedstrijd terug weegt 15% minder
DECAY = 0.85

def compute_weighted_form(results: list) -> float:
    """
    Bereken gewogen vorm uit lijst van resultaten.
    results: lijst van floats, meest recent eerst
             1.0 = gewonnen, 0.5 = gelijk, 0.0 = verloren
    """
    if not results:
        return 0.55  # neutrale prior

    weights = [DECAY ** i for i in range(len(results))]
    weighted_sum = sum(r * w for r, w in zip(results, weights))
    total_weight = sum(weights)

    return round(weighted_sum / total_weight, 4)

def get_team_results(team: str, year: int) -> list:
    """
    Haal alle resultaten op voor een team in een specifiek WK.
    Geeft lijst terug van resultaten, meest recent eerst.
    """
    path = DATA_DIR / f"wc{year}.json"
    if not path.exists():
        return []

    data    = json.loads(path.read_text())
    results = []

    for match in data.get("matches", []):
        team1 = match.get("team1", "")
        team2 = match.get("team2", "")
        score = match.get("score", {})

        if team not in [team1, team2]:
            continue
        if not score:
            continue

        ft = score.get("ft", None)
        if not ft or len(ft) < 2:
            continue

        g1, g2 = int(ft[0]), int(ft[1])

        if team == team1:
            if g1 > g2:   results.append(1.0)
            elif g1 == g2: results.append(0.5)
            else:          results.append(0.0)
        else:
            if g2 > g1:   results.append(1.0)
            elif g1 == g2: results.append(0.5)
            else:          results.append(0.0)

    return list(reversed(results))  # meest recent eerst

def build_form_ratings() -> dict:
    """
    Bouw gewogen vormrating voor alle teams op basis van
    WK 2014, 2018, 2022 — recentere toernooien wegen zwaarder.
    """
    teams_df = pd.read_csv(DATA_DIR / "teams.csv")
    form_ratings = {}

    for team in teams_df["team"]:
        all_results = []

        # WK 2022 meest recent, 2014 minst recent
        for year in [2022, 2018, 2014]:
            results = get_team_results(team, year)
            # Oudere toernooien krijgen extra decay
            if year == 2018:
                results = [r * 0.7 for r in results]
            elif year == 2014:
                results = [r * 0.5 for r in results]
            all_results.extend(results)

        if all_results:
            form_ratings[team] = compute_weighted_form(all_results)
        else:
            # Geen historische data: gebruik huidige form uit teams.csv
            row = teams_df[teams_df["team"] == team]
            if not row.empty:
                form_ratings[team] = float(row.iloc[0]["form"])
            else:
                form_ratings[team] = 0.55

    return form_ratings

def update_teams_form():
    """
    Update de form kolom in teams.csv met gewogen historische vorm.
    """
    teams_df     = pd.read_csv(DATA_DIR / "teams.csv")
    form_ratings = build_form_ratings()

    updated = 0
    for idx, row in teams_df.iterrows():
        team = row["team"]
        if team in form_ratings:
            old_form = row["form"]
            new_form = form_ratings[team]
            teams_df.at[idx, "form"] = new_form
            if abs(old_form - new_form) > 0.01:
                updated += 1

    teams_df.to_csv(DATA_DIR / "teams.csv", index=False)
    print(f"✓ Form bijgewerkt voor {updated} teams")
    return teams_df

def get_current_tournament_form(team: str, matches_df: pd.DataFrame) -> float:
    """
    Bereken vorm op basis van reeds gespeelde WK 2026 wedstrijden.
    Combineert met historische vorm via weging.
    """
    played = matches_df[
        ((matches_df["home"] == team) | (matches_df["away"] == team)) &
        (matches_df["played"] == 1)
    ].sort_values("match_id", ascending=False)

    if played.empty:
        return None  # Nog geen wedstrijden gespeeld

    results = []
    for _, row in played.iterrows():
        gh, ga = int(row["home_score"]), int(row["away_score"])
        if row["home"] == team:
            if gh > ga:    results.append(1.0)
            elif gh == ga: results.append(0.5)
            else:          results.append(0.0)
        else:
            if ga > gh:    results.append(1.0)
            elif gh == ga: results.append(0.5)
            else:          results.append(0.0)

    return compute_weighted_form(results)

def get_form_features(
    team_a: str,
    team_b: str,
    matches_df: pd.DataFrame = None
) -> dict:
    """
    Geef vorm-features terug voor twee teams.
    Tijdens toernooi: combineert historische + actuele vorm.
    """
    teams_df = pd.read_csv(DATA_DIR / "teams.csv")

    def get_form(team):
        # Basis historische vorm
        row = teams_df[teams_df["team"] == team]
        base_form = float(row.iloc[0]["form"]) if not row.empty else 0.55

        # Actuele toernooi-vorm
        if matches_df is not None:
            tournament_form = get_current_tournament_form(team, matches_df)
            if tournament_form is not None:
                # 60% toernooi-vorm, 40% historisch
                return round(0.6 * tournament_form + 0.4 * base_form, 4)

        return base_form

    form_a = get_form(team_a)
    form_b = get_form(team_b)

    return {
        "form_a":          form_a,
        "form_b":          form_b,
        "weighted_form_diff": round(form_a - form_b, 4),
        # Momentum: is één team veel beter in vorm?
        "form_dominance":  round(abs(form_a - form_b), 4),
    }

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Gewogen vorm berekenen voor alle teams...\n")

    form_ratings = build_form_ratings()

    # Top 15
    top = sorted(form_ratings.items(), key=lambda x: -x[1])[:15]
    print("Top 15 teams op gewogen historische vorm:")
    for i, (team, form) in enumerate(top, 1):
        bar = "█" * int(form * 20)
        print(f"  {i:>2}. {team:<25} {bar} {form}")

    print("\nVorm updaten in teams.csv...")
    update_teams_form()

    print("\nVorm-features Nederland vs Argentinië:")
    matches_df = pd.read_csv(DATA_DIR / "matches.csv")
    feats = get_form_features("Netherlands", "Argentina", matches_df)
    for k, v in feats.items():
        print(f"  {k:<25}: {v}")