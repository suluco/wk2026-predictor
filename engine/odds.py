"""
Bookmaker-odds module — The Odds API integratie voor WK 2026.

Werking:
  1. Fetch H2H-odds voor alle WK-matches via The Odds API
  2. Gemiddeld over alle beschikbare bookmakers (beter gecalibreerd dan één boek)
  3. Vig-correctie: ruwe implied probs normaliseren zodat ze optellen tot 1.0
  4. Cache opslaan in data/odds_cache.json (6 uur geldig)
  5. Terugval op None als match niet gevonden of API niet beschikbaar

API key instellen:
  export ODDS_API_KEY="jouw_key"
  of zet ODDS_API_KEY=... in een .env bestand naast dit project
"""

import json
import os
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR  = Path(__file__).parent.parent
DATA_DIR  = ROOT_DIR / "data"

# Laad .env uit de projectroot (werkt ook als .env niet bestaat)
load_dotenv(ROOT_DIR / ".env")
CACHE_PATH = DATA_DIR / "odds_cache.json"
API_BASE  = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_fifa_world_cup"
CACHE_TTL_HOURS = 6

# ── Team-naam mapping ─────────────────────────────────────────────────────────
# The Odds API gebruikt soms andere namen dan onze matches.csv.
# Links: onze naam — rechts: wat de API terugstuurt.
OUR_TO_API = {
    "USA":                      "United States",
    "South Korea":              "South Korea",
    "Ivory Coast":              "Ivory Coast",
    "DR Congo":                 "DR Congo",
    "Czechia":                  "Czech Republic",
    "Curaçao":                  "Curacao",
    "Bosnia and Herzegovina":   "Bosnia and Herzegovina",
}

# Omgekeerde mapping voor het vertalen van API-namen terug naar onze namen
API_TO_OUR = {v: k for k, v in OUR_TO_API.items()}


def _our_to_api(name: str) -> str:
    return OUR_TO_API.get(name, name)


def _api_to_our(name: str) -> str:
    return API_TO_OUR.get(name, name)


# ── Vig-correctie ─────────────────────────────────────────────────────────────

def implied_probs(home_odds: float, draw_odds: float, away_odds: float) -> dict:
    """
    Zet decimale odds om naar vig-gecorrigeerde implied kansen.
    Normalisatie verwijdert de bookmaker-marge zodat de kansen optellen tot 1.0.
    """
    raw = {
        "win_a": 1.0 / home_odds,
        "draw":  1.0 / draw_odds,
        "win_b": 1.0 / away_odds,
    }
    total = sum(raw.values())
    return {k: round(v / total, 4) for k, v in raw.items()}


# ── API fetch ─────────────────────────────────────────────────────────────────

def _fetch_from_api(api_key: str) -> list:
    """Haal alle WK-events op van The Odds API (H2H, EU-regio, decimaal)."""
    url = f"{API_BASE}/sports/{SPORT_KEY}/odds/"
    params = {
        "apiKey":      api_key,
        "regions":     "eu",
        "markets":     "h2h",
        "oddsFormat":  "decimal",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _parse_events(events: list) -> dict:
    """
    Verwerk API-response naar een dict keyed by (home_our, away_our).
    Gemiddeld over alle bookmakers voor betere calibratie.
    """
    result = {}

    for event in events:
        home_api = event.get("home_team", "")
        away_api = event.get("away_team", "")
        home_our = _api_to_our(home_api)
        away_our = _api_to_our(away_api)

        # Verzamel odds per uitkomst over alle bookmakers
        home_odds_list, draw_odds_list, away_odds_list = [], [], []

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                h = outcomes.get(home_api)
                d = outcomes.get("Draw")
                a = outcomes.get(away_api)
                if h and d and a:
                    home_odds_list.append(h)
                    draw_odds_list.append(d)
                    away_odds_list.append(a)

        if not home_odds_list:
            continue

        avg_home = sum(home_odds_list) / len(home_odds_list)
        avg_draw = sum(draw_odds_list) / len(draw_odds_list)
        avg_away = sum(away_odds_list) / len(away_odds_list)

        probs = implied_probs(avg_home, avg_draw, avg_away)
        probs["bookmakers"] = len(home_odds_list)
        probs["commence_time"] = event.get("commence_time", "")

        result[(home_our, away_our)] = probs

    return result


# ── Cache ─────────────────────────────────────────────────────────────────────

def _load_cache() -> tuple[dict, bool]:
    """Laad cache. Geeft (data, is_fresh) terug."""
    if not CACHE_PATH.exists():
        return {}, False
    try:
        raw = json.loads(CACHE_PATH.read_text())
        fetched_at = datetime.fromisoformat(raw["fetched_at"])
        age = datetime.now(timezone.utc) - fetched_at
        is_fresh = age < timedelta(hours=CACHE_TTL_HOURS)
        # Sleutels zijn opgeslagen als "Home||Away" strings
        data = {tuple(k.split("||")): v for k, v in raw["odds"].items()}
        return data, is_fresh
    except Exception:
        return {}, False


def _save_cache(odds_dict: dict):
    """Sla odds op als JSON met timestamp."""
    serialisable = {f"{k[0]}||{k[1]}": v for k, v in odds_dict.items()}
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "odds": serialisable,
    }
    CACHE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


# ── Publieke interface ────────────────────────────────────────────────────────

def load_odds(api_key: str = None, force_refresh: bool = False) -> dict:
    """
    Laad odds: cache als die nog vers is, anders API-call.
    Geeft dict terug: {(home, away): {win_a, draw, win_b, bookmakers}}.
    Geeft lege dict terug als API-key ontbreekt of call mislukt.
    """
    if api_key is None:
        api_key = os.environ.get("ODDS_API_KEY", "")

    # Cache proberen
    if not force_refresh:
        cached, is_fresh = _load_cache()
        if is_fresh and cached:
            return cached

    if not api_key:
        # Geen key: terugval op bestaande cache (ook al verlopen)
        cached, _ = _load_cache()
        if cached:
            print("⚠ ODDS_API_KEY niet ingesteld — verlopen cache gebruikt")
        return cached

    try:
        events = _fetch_from_api(api_key)
        odds_dict = _parse_events(events)
        _save_cache(odds_dict)
        n = len(odds_dict)
        print(f"✓ Odds geladen: {n} wedstrijd{'en' if n != 1 else ''} (The Odds API)")
        return odds_dict
    except requests.HTTPError as e:
        print(f"⚠ Odds API fout: {e}")
    except Exception as e:
        print(f"⚠ Odds ophalen mislukt: {e}")

    # Terugval op cache
    cached, _ = _load_cache()
    return cached


def get_match_odds(home: str, away: str, odds_dict: dict) -> dict | None:
    """
    Geef odds-kansen voor één wedstrijd.
    Probeert ook de omgekeerde volgorde (away, home) en vertaalt win_a/win_b dan mee.
    Geeft None als de wedstrijd niet in de odds staat.
    """
    if not odds_dict:
        return None

    # Directe match
    key = (home, away)
    if key in odds_dict:
        return odds_dict[key]

    # Omgekeerd (bookmakers slaan soms home/away anders op)
    key_rev = (away, home)
    if key_rev in odds_dict:
        o = odds_dict[key_rev]
        return {
            "win_a":      o["win_b"],
            "draw":       o["draw"],
            "win_b":      o["win_a"],
            "bookmakers": o.get("bookmakers", 0),
        }

    return None


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        print("Zet ODDS_API_KEY als omgevingsvariabele om te testen.")
    else:
        odds = load_odds(api_key=key, force_refresh=True)
        print(f"\n{len(odds)} wedstrijden met odds:\n")
        for (home, away), probs in sorted(odds.items()):
            print(f"  {home} vs {away}")
            print(f"    {home}: {round(probs['win_a']*100,1)}%  "
                  f"Gelijkspel: {round(probs['draw']*100,1)}%  "
                  f"{away}: {round(probs['win_b']*100,1)}%  "
                  f"({probs['bookmakers']} bookmakers)")
