import pandas as pd
import numpy as np
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Constanten ────────────────────────────────────────────────────────────────

BASE_ELO    = 1500   # Startrating voor elk team
K_FACTOR    = 40     # Hoe snel Elo verandert (WK-wedstrijden wegen zwaar)
HOME_BONUS  = 0      # Neutraal veld op WK, dus 0

# Elo-ratings per maart 2026 — gebaseerd op WK2022 uitslag, Copa America 2024,
# Euro 2024 en recente kwalificatieresultaten. Dit is de beginstand van het toernooi.
# Wordt bijgewerkt via elo_ratings.csv na elke gespeelde WK 2026 wedstrijd.
INITIAL_ELO = {
    # Top tier — WK/continental titels 2022-2025
    "Argentina":    2000,  # WK2022 + Copa America 2024 + 45-wedstrijden ongeslagen
    "Spain":        1970,  # Euro 2024 + Nations League finalist
    "France":       1950,  # WK2022 runner-up + EK2024 SF
    "England":      1920,  # Euro 2024 finalist
    "Brazil":       1930,  # consistent top-5, Copa2024 QF
    "Portugal":     1910,  # Nations League 2024/25, Nations League + WK-kwalificatie top
    "Netherlands":  1900,  # EK2024 SF, sterk in kwalificatie
    "Germany":      1880,  # Euro2024 QF als gastland, solide WK-kwalificatie
    "Belgium":      1860,  # Nations League sterk, maar golden generation aan einde
    "Morocco":      1840,  # WK2022 SF, AFCON 2021 & 2023 deelname, CAF-top
    "Colombia":     1830,  # Copa2024 finalist, CAF-top, Elo-top 10
    "Uruguay":      1820,  # Copa2024 SF, consistent
    "Croatia":      1810,  # WK2022 3e, Nations League finalist 2023
    "Switzerland":  1790,  # WK2022 QF, Euro2024 QF — steevast bovenste helft
    "USA":          1780,  # Gold Cup 2023 winnaar, thuisland WK
    "Mexico":       1780,  # thuisland WK, altijd solide
    "Norway":       1790,  # sterk Nations League, Haaland-generatie
    "Austria":      1760,  # EK2024 R16, goed kwalificatierecord
    "Japan":        1760,  # WK2022 R16 (verrassingsteam), AFC solide
    "Senegal":      1750,  # AFCON 2021/2022 winnaar, WK2022 R16
    "Ivory Coast":  1740,  # AFCON 2023/24 winnaar
    "Canada":       1740,  # thuisland WK, sterk in CONCACAF kwalificatie
    "South Korea":  1730,  # WK2022 R16, sterk Aziatisch kampioenschap
    "Ecuador":      1710,  # WK2022 groepsfase, CONMEBOL-top
    "Scotland":     1710,  # EK2024 deelname, consistent
    "Algeria":      1700,  # AFCON 2019, solide CAF-team
    "Uzbekistan":   1620,  # opkomend AFC-team
    "Egypt":        1690,  # AFCON finalist 2021, CAF-sterk
    "Tunisia":      1650,  # regelmatige WK-deelnemer
    "Paraguay":     1680,  # CONMEBOL-middenklasse
    "Australia":    1670,  # WK2022 R16, AFC-top
    "South Africa": 1650,  # AFCON deelnemer, solide CAF
    "Ghana":        1660,  # WK-aanwezigheid, AFCON deelname
    "Iran":         1690,  # AFC kwalificatie top, WK-veteraan
    "Saudi Arabia": 1640,  # WK2022 (versloeg Argentina in groepsfase!)
    "Cape Verde":   1650,  # AFCON 2023 QF, opkomend
    "DR Congo":     1620,  # AFCON sterk, solide CAF
    "Qatar":        1590,  # WK2022 gastland, AFC-onderste helft
    "Panama":       1610,  # CONCACAF solide
    "Iraq":         1600,  # AFC-middenklasse
    "Jordan":       1590,  # AFC-middenmoot
    "New Zealand":  1590,  # OFC-top
    "Bolivia":      1590,  # CONMEBOL-onderste tier
    "Haiti":        1560,  # CONCACAF
    "Curaçao":      1540,  # CONCACAF, nieuwkomer
    "Jamaica":      1580,  # CONCACAF
    "Suriname":     1560,  # CONMEBOL, debutant
    "New Caledonia":1460,  # OFC-laagst
    # Play-off kwalificanten
    "Bosnia and Herzegovina": 1740,  # UEFA play-off, sterke generatie
    "Sweden":       1800,  # UEFA, consistent Nations League
    "Turkey":       1770,  # Euro2024 QF, sterk
    "Czechia":      1710,  # Euro2024 R16, solide
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

    for year in [2002, 2006, 2010, 2014, 2018, 2022]:
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
    """Laad Elo-ratings uit CSV (bijgewerkt tijdens toernooi), anders INITIAL_ELO."""
    path = DATA_DIR / "elo_ratings.csv"
    if path.exists():
        df = pd.read_csv(path)
        elo = dict(zip(df["team"], df["elo"]))
        # Vul ontbrekende teams aan vanuit INITIAL_ELO
        for team, rating in INITIAL_ELO.items():
            if team not in elo:
                elo[team] = rating
        return elo
    return INITIAL_ELO.copy()

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