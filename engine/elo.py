import pandas as pd
import numpy as np
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Constanten ────────────────────────────────────────────────────────────────

BASE_ELO    = 1500   # Startrating voor elk team
K_FACTOR    = 40     # Hoe snel Elo verandert (WK-wedstrijden wegen zwaar)
HOME_BONUS  = 0      # Neutraal veld op WK, dus 0

# Initiële Elo-ratings gebaseerd op FIFA-ranking + historische prestaties
INITIAL_ELO = {
    "Argentina":    1950,
    "France":       1940,
    "England":      1880,
    "Belgium":      1860,
    "Brazil":       1930,
    "Portugal":     1900,
    "Netherlands":  1870,
    "Spain":        1910,
    "Morocco":      1780,
    "Germany":      1890,
    "Colombia":     1800,
    "Uruguay":      1820,
    "Croatia":      1790,
    "Japan":        1740,
    "Senegal":      1730,
    "USA":          1760,
    "Mexico":       1770,
    "Switzerland":  1760,
    "Iran":         1680,
    "South Korea":  1720,
    "Ecuador":      1700,
    "Canada":       1720,
    "Norway":       1770,
    "Austria":      1730,
    "Algeria":      1680,
    "Australia":    1660,
    "Qatar":        1580,
    "Ivory Coast":  1710,
    "Egypt":        1670,
    "Scotland":     1700,
    "Tunisia":      1640,
    "Paraguay":     1670,
    "South Africa": 1640,
    "New Zealand":  1580,
    "Ghana":        1650,
    "Uzbekistan":   1610,
    "Bolivia":      1580,
    "Panama":       1600,
    "Jordan":       1580,
    "Saudi Arabia": 1620,
    "Cape Verde":   1620,
    "Haiti":        1540,
    "Curaçao":      1520,
    "Iraq":         1590,
    "Jamaica":      1570,
    "DR Congo":     1600,
    "Suriname":     1540,
    "New Caledonia":1440,
}

# ── Elo berekening ────────────────────────────────────────────────────────────

def expected_score(elo_a: float, elo_b: float) -> float:
    """Verwachte score (0-1) voor team A tegen team B."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo_a: float, elo_b: float, score_a: float, k: float = K_FACTOR):
    """
    Update Elo-ratings na een wedstrijd.
    score_a: 1.0 = A wint, 0.5 = gelijkspel, 0.0 = B wint
    """
    exp_a = expected_score(elo_a, elo_b)
    exp_b = 1 - exp_a
    score_b = 1 - score_a

    new_elo_a = elo_a + k * (score_a - exp_a)
    new_elo_b = elo_b + k * (score_b - exp_b)

    return round(new_elo_a, 1), round(new_elo_b, 1)

def result_to_score(goals_a: int, goals_b: int) -> float:
    if goals_a > goals_b:
        return 1.0
    elif goals_a == goals_b:
        return 0.5
    return 0.0

# ── Elo opbouwen vanuit historische WK-data ───────────────────────────────────

def build_elo_from_history() -> dict:
    """
    Verwerk WK 2014, 2018, 2022 om Elo-ratings bij te werken.
    Geeft een dict terug: team -> elo
    """
    elo = INITIAL_ELO.copy()

    for year in [2014, 2018, 2022]:
        path = DATA_DIR / f"wc{year}.json"
        if not path.exists():
            continue

        data = json.loads(path.read_text())

        for match in data.get("matches", []):
            team1 = match.get("team1", "")
            team2 = match.get("team2", "")
            score = match.get("score", {})

            if not score:
                continue

            # Haal FT score op
            ft = score.get("ft", None)
            if not ft or len(ft) < 2:
                continue

            g1, g2 = int(ft[0]), int(ft[1])

            # Zorg dat beide teams in elo dict zitten
            if team1 not in elo:
                elo[team1] = BASE_ELO
            if team2 not in elo:
                elo[team2] = BASE_ELO

            s1 = result_to_score(g1, g2)
            elo[team1], elo[team2] = update_elo(elo[team1], elo[team2], s1)

    return elo

def get_elo(team: str, elo_dict: dict) -> float:
    """Geef Elo-rating terug, of BASE_ELO als team onbekend is."""
    return elo_dict.get(team, BASE_ELO)

def elo_win_probability(elo_a: float, elo_b: float) -> tuple:
    """
    Bereken winkansen op basis van Elo-verschil.
    Returns: (win_a, draw, win_b) als fracties
    """
    exp_a = expected_score(elo_a, elo_b)

    # Gelijkspelkans schatten: hoger bij gelijkwaardige teams
    elo_diff = abs(elo_a - elo_b)
    draw_prob = max(0.10, 0.28 - (elo_diff / 3000))

    win_a = exp_a * (1 - draw_prob)
    win_b = (1 - exp_a) * (1 - draw_prob)

    return round(win_a, 4), round(draw_prob, 4), round(win_b, 4)

def save_elo(elo_dict: dict):
    """Sla Elo-ratings op als CSV."""
    rows = [{"team": t, "elo": e} for t, e in sorted(elo_dict.items(), key=lambda x: -x[1])]
    pd.DataFrame(rows).to_csv(DATA_DIR / "elo_ratings.csv", index=False)
    print(f"✓ elo_ratings.csv opgeslagen ({len(rows)} teams)")

def load_elo() -> dict:
    """Laad Elo-ratings uit CSV, of bouw opnieuw op."""
    path = DATA_DIR / "elo_ratings.csv"
    if path.exists():
        df = pd.read_csv(path)
        return dict(zip(df["team"], df["elo"]))
    return build_elo_from_history()

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Elo opbouwen vanuit WK 2014/2018/2022...")
    elo = build_elo_from_history()
    save_elo(elo)

    print("\nTop 15 Elo-ratings na historische data:")
    top = sorted(elo.items(), key=lambda x: -x[1])[:15]
    for i, (team, rating) in enumerate(top, 1):
        print(f"  {i:>2}. {team:<20} {rating}")

    print("\nVoorbeeldberekening — Nederland vs Argentinië:")
    elo_nl  = get_elo("Netherlands", elo)
    elo_arg = get_elo("Argentina", elo)
    win_nl, draw, win_arg = elo_win_probability(elo_nl, elo_arg)
    print(f"  Elo Nederland  : {elo_nl}")
    print(f"  Elo Argentinië : {elo_arg}")
    print(f"  Win NL         : {round(win_nl*100,1)}%")
    print(f"  Gelijkspel     : {round(draw*100,1)}%")
    print(f"  Win ARG        : {round(win_arg*100,1)}%")