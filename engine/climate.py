import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Stadion data ──────────────────────────────────────────────────────────────

def load_stadiums() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "stadiums.csv")

def get_stadium_for_match(match_id: int, stadiums_df: pd.DataFrame) -> pd.Series:
    """Vind het stadion voor een bepaalde match_id op basis van range."""
    for _, row in stadiums_df.iterrows():
        start, end = map(int, row["match_id_range"].split("-"))
        if start <= match_id <= end:
            return row
    return stadiums_df.iloc[0]  # fallback

# ── Open-Meteo API (gratis, geen key) ────────────────────────────────────────

def fetch_weather(lat: float, lon: float, date: str) -> dict:
    """
    Haal historische of voorspelde weersdata op via Open-Meteo.
    date format: YYYY-MM-DD
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,precipitation_sum,windspeed_10m_max"
            f"&timezone=auto"
            f"&start_date={date}&end_date={date}"
        )
        r = requests.get(url, timeout=5)
        data = r.json()

        daily = data.get("daily", {})
        temp  = daily.get("temperature_2m_max", [None])[0]
        precip = daily.get("precipitation_sum", [None])[0]
        wind  = daily.get("windspeed_10m_max", [None])[0]

        return {
            "temp_max":    float(temp)   if temp   is not None else 25.0,
            "precip_mm":   float(precip) if precip is not None else 0.0,
            "wind_kmh":    float(wind)   if wind   is not None else 10.0,
        }
    except Exception:
        return {"temp_max": 25.0, "precip_mm": 0.0, "wind_kmh": 10.0}

# ── Klimaatcorrectie features ─────────────────────────────────────────────────

def altitude_penalty(altitude_m: float) -> float:
    """
    Hoogtecorrectie: boven 1500m significante impact op uithoudingsvermogen.
    Teams uit lage landen presteren slechter op hoogte.
    Geeft penalty factor terug (0 = geen penalty, 1 = maximale penalty)
    """
    if altitude_m < 500:
        return 0.0
    elif altitude_m < 1500:
        return 0.03
    elif altitude_m < 2000:
        return 0.08
    else:
        return 0.15  # Mexico City (2240m) — significant voordeel voor gewende teams

def heat_penalty(temp_celsius: float) -> float:
    """
    Hittecorrectie: boven 28°C meetbaar effect op intensiteit.
    Europese teams presteren gemiddeld slechter in extreme hitte.
    """
    if temp_celsius < 20:
        return 0.0
    elif temp_celsius < 28:
        return 0.02
    elif temp_celsius < 33:
        return 0.06
    else:
        return 0.10

def climate_advantage(
    team_confederation: str,
    altitude_m: float,
    temp_celsius: float
) -> float:
    """
    Bereken klimaatvoordeel voor een team op basis van confederatie.
    CONCACAF-teams zijn gewend aan hitte en hoogte in Mexico.
    Europese teams hebben nadeel op hoogte en in extreme hitte.
    """
    alt_pen  = altitude_penalty(altitude_m)
    heat_pen = heat_penalty(temp_celsius)

    # Confederatie-aanpassing
    if team_confederation == "CONCACAF":
        # Gewend aan klimaat → minder penalty
        factor = -0.5
    elif team_confederation in ["CONMEBOL"]:
        # Deels gewend (Bogotá = hoogte, tropisch klimaat)
        factor = -0.2
    elif team_confederation in ["CAF"]:
        # Gewend aan hitte, niet aan hoogte
        heat_pen *= 0.3
        factor = 0.0
    else:
        # UEFA, AFC etc. — volledig penalty
        factor = 1.0

    total_penalty = (alt_pen + heat_pen) * factor
    return round(total_penalty, 4)

def get_climate_features(
    match_id: int,
    date: str,
    team_a_conf: str,
    team_b_conf: str,
    stadiums_df: pd.DataFrame,
    use_api: bool = True
) -> dict:
    """
    Geef klimaatfeatures terug voor een wedstrijd.
    """
    stadium = get_stadium_for_match(match_id, stadiums_df)
    alt     = float(stadium["altitude_m"])
    lat     = float(stadium["lat"])
    lon     = float(stadium["lon"])

    # Weer ophalen of statische data gebruiken
    if use_api and date >= "2026-06-01":
        weather = fetch_weather(lat, lon, date)
        temp    = weather["temp_max"]
        precip  = weather["precip_mm"]
        wind    = weather["wind_kmh"]
    else:
        temp   = float(stadium["avg_temp_june"])
        precip = 5.0
        wind   = 15.0

    adv_a = climate_advantage(team_a_conf, alt, temp)
    adv_b = climate_advantage(team_b_conf, alt, temp)

    return {
        "altitude_m":       alt,
        "temp_celsius":     temp,
        "precip_mm":        precip,
        "wind_kmh":         wind,
        "climate_adv_diff": round(adv_a - adv_b, 4),  # positief = voordeel A
        "alt_penalty":      altitude_penalty(alt),
        "heat_penalty":     heat_penalty(temp),
    }

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    stadiums = load_stadiums()

    print("Klimaatfeatures per speellocatie:\n")
    test_cases = [
        (1,  "2026-06-11", "CONCACAF", "CAF"),      # Mexico City: Mexico vs South Africa
        (10, "2026-06-14", "UEFA",     "UEFA"),      # Guadalajara: Netherlands vs Japan
        (91, "2026-06-16", "UEFA",     "CONMEBOL"),  # Miami: France vs Senegal
    ]

    for match_id, date, conf_a, conf_b in test_cases:
        stadium = get_stadium_for_match(match_id, stadiums)
        feats = get_climate_features(match_id, date, conf_a, conf_b, stadiums, use_api=False)
        print(f"  Match {match_id} | {stadium['city']} ({int(feats['altitude_m'])}m, {feats['temp_celsius']}°C)")
        print(f"    Hoogte penalty : {feats['alt_penalty']}")
        print(f"    Hitte penalty  : {feats['heat_penalty']}")
        print(f"    Klimaat voordeel diff (A-B): {feats['climate_adv_diff']}")
        print()