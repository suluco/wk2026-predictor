import math
import pandas as pd
from pathlib import Path
from engine.elo import get_elo, expected_score

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Spelerdata t/m Laatste 32 ─────────────────────────────────────────────────
# Handmatig bijgehouden na de Laatste 32 — geen player-level API beschikbaar.
# Hook voor echte data: vervang TOURNAMENT_SCORERS door een live scraper/feed,
# of verwerk per-speler stats na elke wedstrijd via record_result() in ratings.py.
# goals = totaal over groepsfase + Laatste 32 | minutes = totaal speelminuten
# Alleen teams die door zijn naar de 1/8e finale staan in deze lijst.
TOURNAMENT_SCORERS = [
    # Groepsfase goals + Laatste 32 goals
    {"name": "Kylian Mbappe",     "team": "France",       "goals": 7, "assists": 3, "minutes": 380, "shots": 24},
    {"name": "Erling Haaland",    "team": "Norway",        "goals": 6, "assists": 1, "minutes": 360, "shots": 22},
    {"name": "Vinicius Jr",       "team": "Brazil",        "goals": 5, "assists": 2, "minutes": 360, "shots": 16},
    {"name": "Lionel Messi",      "team": "Argentina",     "goals": 5, "assists": 2, "minutes": 310, "shots": 14},
    {"name": "Romelu Lukaku",     "team": "Belgium",       "goals": 4, "assists": 1, "minutes": 320, "shots": 12},
    {"name": "Harry Kane",        "team": "England",       "goals": 4, "assists": 1, "minutes": 360, "shots": 13},
    {"name": "Cristiano Ronaldo", "team": "Portugal",      "goals": 4, "assists": 1, "minutes": 290, "shots": 11},
    {"name": "Lautaro Martinez",  "team": "Argentina",     "goals": 3, "assists": 2, "minutes": 335, "shots": 10},
    {"name": "Raphinha",          "team": "Brazil",        "goals": 3, "assists": 2, "minutes": 360, "shots": 10},
    {"name": "Bukayo Saka",       "team": "England",       "goals": 3, "assists": 2, "minutes": 360, "shots": 11},
    {"name": "Pedri",             "team": "Spain",         "goals": 3, "assists": 3, "minutes": 360, "shots": 8},
    {"name": "Ferran Torres",     "team": "Spain",         "goals": 3, "assists": 1, "minutes": 280, "shots": 9},
    {"name": "James Rodriguez",   "team": "Colombia",      "goals": 3, "assists": 2, "minutes": 360, "shots": 8},
    {"name": "Youssef En-Nesyri", "team": "Morocco",       "goals": 3, "assists": 1, "minutes": 360, "shots": 9},
    {"name": "Jonathan David",    "team": "Canada",        "goals": 2, "assists": 2, "minutes": 360, "shots": 8},
    {"name": "Rafael Leao",       "team": "Portugal",      "goals": 2, "assists": 2, "minutes": 310, "shots": 9},
    {"name": "Breel Embolo",      "team": "Switzerland",   "goals": 2, "assists": 1, "minutes": 360, "shots": 7},
    {"name": "Miguel Almiron",    "team": "Paraguay",      "goals": 2, "assists": 1, "minutes": 360, "shots": 6},
]

# Legacy alias zodat bestaande code die GROUP_STAGE_SCORERS importeert blijft werken
GROUP_STAGE_SCORERS = TOURNAMENT_SCORERS

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

    # Gebruik de eerstvolgende niet-gespeelde KO-ronde als basis voor next_opponent
    for stage in ("R16", "QF", "SF", "F", "R32"):
        upcoming = matches_df[
            (matches_df["stage"] == stage) &
            (matches_df["played"] == 0)
        ]
        if not upcoming.empty:
            current_stage = stage
            current_round = upcoming
            break
    else:
        current_stage = "R32"
        current_round = matches_df[matches_df["stage"] == "R32"]

    # Volgende tegenstander per team op basis van huidige ronde
    next_opponent: dict = {}
    for _, row in current_round.iterrows():
        next_opponent[str(row["home"])] = str(row["away"])
        next_opponent[str(row["away"])] = str(row["home"])

    # Gemiddeld Elo van alle deelnemers in de huidige ronde als baseline
    current_teams = set(current_round["home"].tolist() + current_round["away"].tolist())
    avg_elo = (
        sum(get_elo(t, elo_dict) for t in current_teams) / len(current_teams)
        if current_teams else 1600.0
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
        # Maximaal 4 wedstrijden vanaf R32, 3 vanaf R16, etc.
        rounds_left = {"R32": 5, "R16": 4, "QF": 3, "SF": 2, "F": 1}.get(current_stage, 4)
        adv_prob    = expected_score(get_elo(team, elo_dict), avg_elo)
        exp_matches = round(sum(adv_prob ** i for i in range(rounds_left)), 1)

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
