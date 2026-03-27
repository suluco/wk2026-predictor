import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Haversine afstand ─────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Bereken afstand in km tussen twee coördinaten."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return round(2 * R * atan2(sqrt(a), sqrt(1-a)), 1)

# ── Tijdzone offset ───────────────────────────────────────────────────────────

TIMEZONE_OFFSETS = {
    "America/Mexico_City": -6,
    "America/Chicago":     -5,
    "America/New_York":    -4,
    "America/Los_Angeles": -7,
    "America/Vancouver":   -7,
}

TEAM_HOME_TIMEZONE = {
    # UTC offset van thuisland
    "France": 2, "Germany": 2, "Spain": 2, "Netherlands": 2,
    "Belgium": 2, "Portugal": 2, "England": 1, "Croatia": 2,
    "Switzerland": 2, "Austria": 2, "Norway": 2, "Denmark": 2,
    "Scotland": 1, "Italy": 2, "Sweden": 2, "Poland": 2,
    "Turkey": 3, "Kosovo": 1, "Czechia": 2, "Bosnia and Herzegovina": 2,
    "Brazil": -3, "Argentina": -3, "Uruguay": -3, "Colombia": -5,
    "Ecuador": -5, "Paraguay": -4,
    "USA": -5, "Mexico": -6, "Canada": -5, "Panama": -5,
    "Curaçao": -4, "Haiti": -5, "Jamaica": -5,
    "Morocco": 1, "Senegal": 0, "Ivory Coast": 0, "Ghana": 0,
    "Egypt": 2, "Tunisia": 1, "South Africa": 2, "Cape Verde": -1,
    "Japan": 9, "South Korea": 9, "Iran": 3.5, "Saudi Arabia": 3,
    "Australia": 10, "New Zealand": 12, "Uzbekistan": 5,
    "Qatar": 3, "Jordan": 3, "Iraq": 3,
}

def jetlag_hours(team: str, venue_timezone: str) -> float:
    """
    Bereken tijdzoneverschil als proxy voor jetlag.
    """
    home_tz  = TEAM_HOME_TIMEZONE.get(team, 0)
    venue_tz = TIMEZONE_OFFSETS.get(venue_timezone, -5)
    return abs(home_tz - venue_tz)

# ── Stadion coördinaten lookup ────────────────────────────────────────────────

def get_stadium_coords(match_id: int, stadiums_df: pd.DataFrame) -> tuple:
    for _, row in stadiums_df.iterrows():
        start, end = map(int, row["match_id_range"].split("-"))
        if start <= match_id <= end:
            return float(row["lat"]), float(row["lon"]), row["timezone"]
    return 25.0, -100.0, "America/Mexico_City"

# ── Vermoeidheidsfeatures per team ────────────────────────────────────────────

def get_fatigue_features(
    team: str,
    match_id: int,
    match_date: str,
    matches_df: pd.DataFrame,
    stadiums_df: pd.DataFrame,
) -> dict:
    """
    Bereken vermoeidheidsfeatures voor een team voor een specifieke wedstrijd.

    Features:
      - days_rest: dagen rust voor deze wedstrijd
      - travel_km: afstand gereisd van vorige wedstrijd
      - total_travel_km: totaal gereisd in toernooi
      - jetlag_hours: tijdzoneverschil thuisland vs speelstad
      - fatigue_score: gecombineerde vermoeidheidscore (0=fit, 1=uitgeput)
    """
    # Huidige wedstrijd stadion
    cur_lat, cur_lon, cur_tz = get_stadium_coords(match_id, stadiums_df)

    # Vind vorige wedstrijden van dit team
    prev_matches = matches_df[
        ((matches_df["home"] == team) | (matches_df["away"] == team)) &
        (matches_df["match_id"] < match_id) &
        (matches_df["played"] >= 0)
    ].sort_values("match_id")

    # Rustdagen
    if len(prev_matches) == 0:
        days_rest   = 10  # Eerste wedstrijd: vol rust
        travel_km   = 0
        total_travel = 0
    else:
        last_match   = prev_matches.iloc[-1]
        last_date    = pd.to_datetime(last_match["date"])
        cur_date     = pd.to_datetime(match_date)
        days_rest    = (cur_date - last_date).days

        # Reisafstand van vorige wedstrijd
        prev_lat, prev_lon, _ = get_stadium_coords(int(last_match["match_id"]), stadiums_df)
        travel_km = haversine_km(prev_lat, prev_lon, cur_lat, cur_lon)

        # Totaal gereisd
        total_travel = travel_km
        for i in range(len(prev_matches) - 1):
            m1 = prev_matches.iloc[i]
            m2 = prev_matches.iloc[i+1]
            lat1, lon1, _ = get_stadium_coords(int(m1["match_id"]), stadiums_df)
            lat2, lon2, _ = get_stadium_coords(int(m2["match_id"]), stadiums_df)
            total_travel += haversine_km(lat1, lon1, lat2, lon2)

    # Jetlag
    jl = jetlag_hours(team, cur_tz)

    # Gecombineerde vermoeidheidscore
    rest_score    = max(0, (7 - days_rest) / 7)       # minder rust = meer moe
    travel_score  = min(travel_km / 3000, 1.0)         # lang reizen = meer moe
    jetlag_score  = min(jl / 12, 1.0)                  # groot tijdverschil = meer moe
    fatigue_score = round(0.4 * rest_score + 0.35 * travel_score + 0.25 * jetlag_score, 4)

    return {
        "days_rest":      days_rest,
        "travel_km":      round(travel_km, 1),
        "total_travel_km": round(total_travel, 1),
        "jetlag_hours":   jl,
        "fatigue_score":  fatigue_score,
    }

def get_fatigue_diff(
    home: str,
    away: str,
    match_id: int,
    match_date: str,
    matches_df: pd.DataFrame,
    stadiums_df: pd.DataFrame,
) -> dict:
    """
    Geef vermoeidheidsfeatures terug als verschil (home - away).
    Positief = home meer moe, negatief = away meer moe.
    """
    f_home = get_fatigue_features(home, match_id, match_date, matches_df, stadiums_df)
    f_away = get_fatigue_features(away, match_id, match_date, matches_df, stadiums_df)

    return {
        "days_rest_home":     f_home["days_rest"],
        "days_rest_away":     f_away["days_rest"],
        "days_rest_diff":     f_home["days_rest"] - f_away["days_rest"],
        "travel_km_diff":     f_home["travel_km"] - f_away["travel_km"],
        "total_travel_diff":  f_home["total_travel_km"] - f_away["total_travel_km"],
        "jetlag_diff":        f_home["jetlag_hours"] - f_away["jetlag_hours"],
        "fatigue_diff":       round(f_home["fatigue_score"] - f_away["fatigue_score"], 4),
    }

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    matches_df  = pd.read_csv(DATA_DIR / "matches.csv")
    stadiums_df = pd.read_csv(DATA_DIR / "stadiums.csv")

    print("Vermoeidheidsfeatures — eerste wedstrijden:\n")

    tests = [
        ("Netherlands", "Japan",    10, "2026-06-14"),
        ("France",      "Senegal",  17, "2026-06-16"),
        ("Argentina",   "Algeria",  19, "2026-06-17"),
    ]

    for home, away, mid, date in tests:
        feats = get_fatigue_diff(home, away, mid, date, matches_df, stadiums_df)
        print(f"  {home} vs {away} (match {mid})")
        print(f"    Rustdagen    : {feats['days_rest_home']}d vs {feats['days_rest_away']}d")
        print(f"    Reisafstand  : {feats['travel_km_diff']} km diff")
        print(f"    Jetlag       : {feats['jetlag_diff']} uur diff")
        print(f"    Fatigue diff : {feats['fatigue_diff']}")
        print()

    print("Afstandstest — Mexico City naar Miami:")
    print(f"  {haversine_km(19.3029, -99.1505, 25.9580, -80.2389)} km")