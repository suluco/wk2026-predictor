import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Statische squad data ──────────────────────────────────────────────────────
# Gebaseerd op FIFA 25 ratings + Champions League presentie + sterspelers
# avg_rating: gemiddelde FIFA rating van de verwachte basiself
# ucl_players: aantal spelers actief in Champions League (proxy voor topniveau)
# star_player: heeft het team een speler in de wereldtop 10 (bool)
# coach_rating: subjectieve coachkwaliteit 1-10 op basis van historische prestaties

SQUAD_DATA = {
    "Argentina":   {"avg_rating": 85, "ucl_players": 10, "star_player": 1, "coach_rating": 9.0, "avg_age": 27.5},
    "France":      {"avg_rating": 85, "ucl_players": 10, "star_player": 1, "coach_rating": 8.0, "avg_age": 26.8},
    "Spain":       {"avg_rating": 85, "ucl_players": 10, "star_player": 1, "coach_rating": 8.5, "avg_age": 25.2},
    "England":     {"avg_rating": 83, "ucl_players": 9,  "star_player": 1, "coach_rating": 7.5, "avg_age": 26.5},
    "Belgium":     {"avg_rating": 81, "ucl_players": 7,  "star_player": 0, "coach_rating": 7.0, "avg_age": 29.5},
    "Brazil":      {"avg_rating": 83, "ucl_players": 9,  "star_player": 1, "coach_rating": 7.5, "avg_age": 26.2},
    "Portugal":    {"avg_rating": 83, "ucl_players": 8,  "star_player": 1, "coach_rating": 7.8, "avg_age": 27.8},
    "Netherlands": {"avg_rating": 82, "ucl_players": 8,  "star_player": 1, "coach_rating": 8.0, "avg_age": 26.5},
    "Morocco":     {"avg_rating": 78, "ucl_players": 5,  "star_player": 0, "coach_rating": 8.0, "avg_age": 27.0},
    "Germany":     {"avg_rating": 82, "ucl_players": 8,  "star_player": 0, "coach_rating": 7.5, "avg_age": 26.0},
    "Colombia":    {"avg_rating": 79, "ucl_players": 4,  "star_player": 0, "coach_rating": 7.5, "avg_age": 27.5},
    "Uruguay":     {"avg_rating": 79, "ucl_players": 4,  "star_player": 0, "coach_rating": 7.5, "avg_age": 28.5},
    "Croatia":     {"avg_rating": 79, "ucl_players": 4,  "star_player": 0, "coach_rating": 7.8, "avg_age": 30.2},
    "Japan":       {"avg_rating": 76, "ucl_players": 3,  "star_player": 0, "coach_rating": 7.5, "avg_age": 26.8},
    "Senegal":     {"avg_rating": 77, "ucl_players": 3,  "star_player": 0, "coach_rating": 7.0, "avg_age": 27.2},
    "USA":         {"avg_rating": 77, "ucl_players": 4,  "star_player": 0, "coach_rating": 7.0, "avg_age": 25.5},
    "Mexico":      {"avg_rating": 77, "ucl_players": 2,  "star_player": 0, "coach_rating": 7.0, "avg_age": 27.8},
    "Switzerland": {"avg_rating": 78, "ucl_players": 4,  "star_player": 0, "coach_rating": 7.5, "avg_age": 28.0},
    "Iran":        {"avg_rating": 73, "ucl_players": 1,  "star_player": 0, "coach_rating": 6.5, "avg_age": 28.5},
    "South Korea": {"avg_rating": 76, "ucl_players": 3,  "star_player": 0, "coach_rating": 7.0, "avg_age": 27.0},
    "Ecuador":     {"avg_rating": 74, "ucl_players": 1,  "star_player": 0, "coach_rating": 6.8, "avg_age": 26.5},
    "Canada":      {"avg_rating": 75, "ucl_players": 3,  "star_player": 0, "coach_rating": 7.0, "avg_age": 25.8},
    "Norway":      {"avg_rating": 78, "ucl_players": 4,  "star_player": 1, "coach_rating": 7.0, "avg_age": 25.5},
    "Austria":     {"avg_rating": 77, "ucl_players": 4,  "star_player": 0, "coach_rating": 7.2, "avg_age": 27.0},
    "Algeria":     {"avg_rating": 74, "ucl_players": 2,  "star_player": 0, "coach_rating": 6.8, "avg_age": 27.5},
    "Australia":   {"avg_rating": 73, "ucl_players": 1,  "star_player": 0, "coach_rating": 6.8, "avg_age": 27.5},
    "Qatar":       {"avg_rating": 68, "ucl_players": 0,  "star_player": 0, "coach_rating": 6.5, "avg_age": 26.0},
    "Ivory Coast": {"avg_rating": 76, "ucl_players": 3,  "star_player": 0, "coach_rating": 7.0, "avg_age": 27.8},
    "Egypt":       {"avg_rating": 74, "ucl_players": 1,  "star_player": 0, "coach_rating": 6.8, "avg_age": 28.0},
    "Scotland":    {"avg_rating": 75, "ucl_players": 2,  "star_player": 0, "coach_rating": 7.0, "avg_age": 27.5},
    "Tunisia":     {"avg_rating": 72, "ucl_players": 1,  "star_player": 0, "coach_rating": 6.5, "avg_age": 27.8},
    "Paraguay":    {"avg_rating": 73, "ucl_players": 1,  "star_player": 0, "coach_rating": 6.8, "avg_age": 27.0},
    "South Africa":{"avg_rating": 72, "ucl_players": 0,  "star_player": 0, "coach_rating": 6.5, "avg_age": 27.5},
    "New Zealand": {"avg_rating": 70, "ucl_players": 0,  "star_player": 0, "coach_rating": 6.5, "avg_age": 27.0},
    "Ghana":       {"avg_rating": 73, "ucl_players": 1,  "star_player": 0, "coach_rating": 6.8, "avg_age": 26.5},
    "Uzbekistan":  {"avg_rating": 70, "ucl_players": 0,  "star_player": 0, "coach_rating": 6.5, "avg_age": 26.0},
    "Panama":      {"avg_rating": 70, "ucl_players": 0,  "star_player": 0, "coach_rating": 6.5, "avg_age": 28.0},
    "Jordan":      {"avg_rating": 68, "ucl_players": 0,  "star_player": 0, "coach_rating": 6.2, "avg_age": 27.0},
    "Saudi Arabia":{"avg_rating": 71, "ucl_players": 0,  "star_player": 0, "coach_rating": 6.8, "avg_age": 27.5},
    "Cape Verde":  {"avg_rating": 71, "ucl_players": 1,  "star_player": 0, "coach_rating": 6.8, "avg_age": 27.0},
    "Haiti":       {"avg_rating": 66, "ucl_players": 0,  "star_player": 0, "coach_rating": 6.0, "avg_age": 26.5},
    "Curaçao":     {"avg_rating": 65, "ucl_players": 0,  "star_player": 0, "coach_rating": 6.0, "avg_age": 27.0},
    "Iraq":        {"avg_rating": 68, "ucl_players": 0,  "star_player": 0, "coach_rating": 6.2, "avg_age": 26.5},
    "Jamaica":     {"avg_rating": 69, "ucl_players": 0,  "star_player": 0, "coach_rating": 6.3, "avg_age": 26.0},
    "DR Congo":    {"avg_rating": 70, "ucl_players": 1,  "star_player": 0, "coach_rating": 6.5, "avg_age": 26.5},
    "Bolivia":     {"avg_rating": 67, "ucl_players": 0,  "star_player": 0, "coach_rating": 6.2, "avg_age": 27.0},
    "New Caledonia":{"avg_rating": 60, "ucl_players": 0, "star_player": 0, "coach_rating": 5.5, "avg_age": 27.0},
    "Sweden":      {"avg_rating": 78, "ucl_players": 4,  "star_player": 1, "coach_rating": 7.2, "avg_age": 26.5},
    "Poland":      {"avg_rating": 77, "ucl_players": 3,  "star_player": 1, "coach_rating": 7.0, "avg_age": 29.0},
    "Turkey":      {"avg_rating": 77, "ucl_players": 3,  "star_player": 0, "coach_rating": 7.0, "avg_age": 27.5},
    "Kosovo":      {"avg_rating": 72, "ucl_players": 1,  "star_player": 0, "coach_rating": 6.5, "avg_age": 26.5},
    "Denmark":     {"avg_rating": 79, "ucl_players": 5,  "star_player": 0, "coach_rating": 7.5, "avg_age": 27.0},
    "Czechia":     {"avg_rating": 75, "ucl_players": 2,  "star_player": 0, "coach_rating": 7.0, "avg_age": 27.5},
    "Italy":       {"avg_rating": 80, "ucl_players": 6,  "star_player": 0, "coach_rating": 7.8, "avg_age": 27.8},
    "Bosnia and Herzegovina": {"avg_rating": 74, "ucl_players": 2, "star_player": 0, "coach_rating": 7.0, "avg_age": 27.0},
    "Suriname":    {"avg_rating": 68, "ucl_players": 0,  "star_player": 0, "coach_rating": 6.2, "avg_age": 26.5},
    "TBD":         {"avg_rating": 73, "ucl_players": 1,  "star_player": 0, "coach_rating": 6.8, "avg_age": 27.0},
}

def get_squad(team: str) -> dict:
    return SQUAD_DATA.get(team, SQUAD_DATA["TBD"])

def get_squad_features(team_a: str, team_b: str) -> dict:
    """
    Geef squad-features terug als verschil (A - B).
    """
    a = get_squad(team_a)
    b = get_squad(team_b)

    # Squadsterkte score (gewogen combinatie)
    def strength_score(s):
        return (
            s["avg_rating"] * 0.40 +
            s["ucl_players"] * 1.5 +
            s["star_player"] * 5.0 +
            s["coach_rating"] * 1.0
        )

    score_a = strength_score(a)
    score_b = strength_score(b)

    # Leeftijdspiek: 26-28 jaar is optimaal
    def age_factor(age):
        if 26 <= age <= 28:
            return 1.0
        elif 24 <= age < 26 or 28 < age <= 30:
            return 0.95
        else:
            return 0.88

    return {
        "squad_rating_diff":  round(a["avg_rating"] - b["avg_rating"], 1),
        "ucl_players_diff":   a["ucl_players"] - b["ucl_players"],
        "star_player_diff":   a["star_player"] - b["star_player"],
        "coach_rating_diff":  round(a["coach_rating"] - b["coach_rating"], 2),
        "squad_score_diff":   round(score_a - score_b, 2),
        "age_factor_diff":    round(age_factor(a["avg_age"]) - age_factor(b["avg_age"]), 4),
    }

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Netherlands", "Argentina"),
        ("France",      "Brazil"),
        ("Germany",     "Japan"),
        ("Morocco",     "Spain"),
    ]

    print("Squad strength features:\n")
    for a, b in tests:
        feats = get_squad_features(a, b)
        winner = a if feats["squad_score_diff"] > 0 else b
        print(f"  {a} vs {b}")
        print(f"    Rating diff    : {feats['squad_rating_diff']}")
        print(f"    UCL spelers    : {feats['ucl_players_diff']}")
        print(f"    Sterspeler     : {feats['star_player_diff']}")
        print(f"    Coach diff     : {feats['coach_rating_diff']}")
        print(f"    Squad score    : {feats['squad_score_diff']} → sterkere squad: {winner}")
        print()