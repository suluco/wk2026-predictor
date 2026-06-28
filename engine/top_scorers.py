import math
import pandas as pd
from pathlib import Path
from engine.elo import get_elo, expected_score

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Spelerdata groepsfase ─────────────────────────────────────────────────────
# Handmatig bijgehouden na de groepsfase — geen player-level API beschikbaar.
# Hook voor echte data: vervang GROUP_STAGE_SCORERS door een live scraper/feed,
# of verwerk per-speler stats na elke wedstrijd via record_result() in ratings.py.
# Kolommen: name, team, goals, assists, minutes, shots (shots = proxy voor volume)
GROUP_STAGE_SCORERS = [
    {"name": "Kylian Mbappe",     "team": "France",      "goals": 5, "assists": 2, "minutes": 270, "shots": 18},
    {"name": "Erling Haaland",    "team": "Norway",       "goals": 4, "assists": 1, "minutes": 270, "shots": 16},
    {"name": "Florian Wirtz",     "team": "Germany",      "goals": 4, "assists": 2, "minutes": 270, "shots": 12},
    {"name": "Sadio Mane",        "team": "Senegal",      "goals": 4, "assists": 1, "minutes": 270, "shots": 11},
    {"name": "Jamal Musiala",     "team": "Germany",      "goals": 3, "assists": 3, "minutes": 270, "shots": 10},
    {"name": "Cody Gakpo",        "team": "Netherlands",  "goals": 3, "assists": 2, "minutes": 270, "shots": 9},
    {"name": "Lionel Messi",      "team": "Argentina",    "goals": 3, "assists": 2, "minutes": 215, "shots": 9},
    {"name": "Vinicius Jr",       "team": "Brazil",       "goals": 3, "assists": 1, "minutes": 270, "shots": 11},
    {"name": "Romelu Lukaku",     "team": "Belgium",      "goals": 3, "assists": 1, "minutes": 230, "shots": 8},
    {"name": "Xavi Simons",       "team": "Netherlands",  "goals": 2, "assists": 2, "minutes": 250, "shots": 7},
    {"name": "Harry Kane",        "team": "England",      "goals": 2, "assists": 1, "minutes": 270, "shots": 9},
    {"name": "Cristiano Ronaldo", "team": "Portugal",     "goals": 2, "assists": 1, "minutes": 200, "shots": 7},
    {"name": "Bukayo Saka",       "team": "England",      "goals": 2, "assists": 2, "minutes": 270, "shots": 8},
    {"name": "Lautaro Martinez",  "team": "Argentina",    "goals": 2, "assists": 1, "minutes": 245, "shots": 7},
    {"name": "Rafael Leao",       "team": "Portugal",     "goals": 2, "assists": 2, "minutes": 230, "shots": 8},
]

# ELO_ALPHA komt overeen met de waarde in simulator.py (Elo-gewogen lambda-schaling)
_ELO_ALPHA = 0.45


def predict_top_scorers(resources: dict, top_n: int = 10) -> list:
    """
    Voorspel topscorers voor de rest van het toernooi.

    Per speler:
      - goals/90 uit de groepsfase als scoring-rate λ
      - score_probability_per_match = P(scoort ≥ 1) via Poisson (1 - e^-λ),
        gecorrigeerd voor tegenstander-sterkte via Elo (zelfde formule als simulator.py)
      - matches_remaining = verwacht aantal resterende wedstrijden via Elo-winkans
        per ronde (geometrische som over max 5 ronden: R32 t/m finale)

    Geeft een gesorteerde lijst van dicts terug met de velden die de spec vereist:
      name, team, goals_so_far, matches_remaining, score_probability_per_match
    Plus aanvullende context: assists, goals_per_90, next_opponent, projected_goals.
    """
    elo_dict = resources["elo_dict"]

    matches_df = pd.read_csv(DATA_DIR / "matches.csv")
    r32 = matches_df[matches_df["stage"] == "R32"]

    # Volgende tegenstander per team op basis van R32-schema
    next_opponent: dict = {}
    for _, row in r32.iterrows():
        next_opponent[row["home"]] = row["away"]
        next_opponent[row["away"]] = row["home"]

    # Gemiddeld Elo van alle R32-deelnemers als baseline per ronde
    r32_teams = set(r32["home"].tolist() + r32["away"].tolist())
    avg_elo = (
        sum(get_elo(t, elo_dict) for t in r32_teams) / len(r32_teams)
        if r32_teams else 1600.0
    )

    results = []
    for player in GROUP_STAGE_SCORERS:
        team  = player["team"]
        goals = player["goals"]
        mins  = player["minutes"]

        # λ per 90 minuten
        nineties = mins / 90.0
        lam_90   = goals / nineties if nineties > 0 else 0.0

        # Tegenstander in R32
        opp = next_opponent.get(team)

        # Elo-aanpassing voor de R32-tegenstander (spiegelt simulator.py:85-89)
        if opp:
            team_elo = get_elo(team, elo_dict)
            opp_elo  = get_elo(opp, elo_dict)
            exp      = expected_score(team_elo, opp_elo)  # P(team wint), 0-1
            # Zwakke tegenstander → exp > 0.5 → factor > 1 → hogere λ
            lam_adj = lam_90 * (exp / 0.5) ** _ELO_ALPHA
        else:
            # Team niet in R32: geen wedstrijden meer te verwachten
            results.append({
                "name":                       player["name"],
                "team":                       team,
                "goals_so_far":               goals,
                "assists_so_far":             player.get("assists", 0),
                "goals_per_90":               round(lam_90, 3),
                "matches_remaining":          0,
                "score_probability_per_match": 0.0,
                "next_opponent":              "—",
                "projected_goals":            0.0,
            })
            continue

        # P(scoort ≥ 1 in de R32-wedstrijd) — dit is de spec-eis
        score_prob = round((1 - math.exp(-lam_adj)) * 100, 1)

        # Verwachte resterende wedstrijden via Elo-winkans per ronde
        # adv_prob = kans op doorgaan in iedere volgende ronde
        adv_prob    = expected_score(get_elo(team, elo_dict), avg_elo)
        # exp_matches = R32 (zeker) + R16 * p + QF * p² + SF * p³ + Finale * p⁴
        exp_matches = round(sum(adv_prob ** i for i in range(5)), 1)

        results.append({
            "name":                       player["name"],
            "team":                       team,
            "goals_so_far":               goals,
            "assists_so_far":             player.get("assists", 0),
            "goals_per_90":               round(lam_90, 3),
            "matches_remaining":          exp_matches,
            "score_probability_per_match": score_prob,
            "next_opponent":              opp,
            "projected_goals":            round(lam_adj * exp_matches, 2),
        })

    results.sort(key=lambda x: (-x["projected_goals"], -x["goals_so_far"]))
    return results[:top_n]
