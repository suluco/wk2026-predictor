"""
auto_updater.py
===============
Draai dit script dagelijks tijdens het toernooi:
    python -m engine.auto_updater

Het script:
1. Haalt de laatste WK-uitslagen op via openfootball GitHub
2. Vult TBD-teams in zodra play-off winnaars bekend zijn
3. Verwerkt nieuwe uitslagen via record_result (Elo + teamsterktes updaten)
4. Hertraint het ML-model
"""

import requests
import json
import pandas as pd
from pathlib import Path
from datetime import date

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Play-off team mapping ─────────────────────────────────────────────────────
# openfootball gebruikt deze namen voor TBD-teams
PLAYOFF_MAPPING = {
    "UEFA Path A winner": {"group": "B"},
    "UEFA Path B winner": {"group": "F"},
    "UEFA Path C winner": {"group": "D"},
    "UEFA Path D winner": {"group": "A"},
    "IC Path 1 winner":   {"group": "K"},
    "IC Path 2 winner":   {"group": "I"},
}

# Bekende naam-aliassen tussen openfootball en onze teams.csv
NAME_ALIASES = {
    "Korea Republic":      "South Korea",
    "IR Iran":             "Iran",
    "Côte d'Ivoire":       "Ivory Coast",
    "USA":                 "USA",
    "Türkiye":             "Turkey",
    "Bosnia-Herzegovina":  "Bosnia and Herzegovina",
    "Bosnia & Herzegovina":"Bosnia and Herzegovina",
    "Czech Republic":      "Czechia",
}

def normalize_name(name: str) -> str:
    return NAME_ALIASES.get(name, name)

# ── Openfootball WK 2026 data ophalen ────────────────────────────────────────

def fetch_wc2026() -> dict:
    url = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"⚠️  GitHub returned {r.status_code} — mogelijk nog geen 2026 data")
            return {}
    except Exception as e:
        print(f"⚠️  Kon data niet ophalen: {e}")
        return {}

# ── TBD teams invullen ────────────────────────────────────────────────────────

def update_tbds_from_data(wc_data: dict):
    """
    Vul TBD-teams in matches.csv in op basis van openfootball data.
    """
    matches_df = pd.read_csv(DATA_DIR / "matches.csv")
    updated    = 0

    if not wc_data:
        return

    for match in wc_data.get("matches", []):
        team1 = normalize_name(match.get("team1", ""))
        team2 = normalize_name(match.get("team2", ""))
        group = match.get("group", "").replace("Group ", "")

        if not team1 or not team2 or not group:
            continue

        # Zoek rijen met TBD in deze groep
        mask = (matches_df["group"] == group)

        for idx, row in matches_df[mask].iterrows():
            if row["home"] == "TBD" and team1 not in ["TBD", ""]:
                matches_df.at[idx, "home"] = team1
                updated += 1
            if row["away"] == "TBD" and team2 not in ["TBD", ""]:
                matches_df.at[idx, "away"] = team2
                updated += 1

    if updated > 0:
        matches_df.to_csv(DATA_DIR / "matches.csv", index=False)
        print(f"✓ {updated} TBD-teams ingevuld in matches.csv")
    else:
        print("  Geen nieuwe TBD-teams gevonden")

# ── Nieuwe uitslagen verwerken ────────────────────────────────────────────────

def sync_results(wc_data: dict):
    """
    Vergelijk openfootball uitslagen met matches.csv.
    Verwerk nieuwe uitslagen automatisch.
    """
    if not wc_data:
        return

    matches_df = pd.read_csv(DATA_DIR / "matches.csv")
    # FIX 5: importeer ook retrain_model zodat we het model eenmalig hertrainen
    # na de volledige batch in plaats van na elke individuele uitslag.
    from engine.ratings import record_result, retrain_model

    new_results = 0

    for match in wc_data.get("matches", []):
        score = match.get("score", {})
        if not score:
            continue

        ft = score.get("ft", None)
        if not ft or len(ft) < 2:
            continue

        team1 = normalize_name(match.get("team1", ""))
        team2 = normalize_name(match.get("team2", ""))
        g1, g2 = int(ft[0]), int(ft[1])

        # Zoek wedstrijd in matches.csv
        mask = (
            (matches_df["home"] == team1) &
            (matches_df["away"] == team2) &
            (matches_df["played"] == 0)
        )

        if mask.any():
            match_id = int(matches_df[mask].iloc[0]["match_id"])
            print(f"  Nieuwe uitslag: {team1} {g1}–{g2} {team2} (match {match_id})")
            # FIX 5: retrain=False — sla ML-hertraining over tijdens de batch;
            # was retrain=True (default), waardoor het model N keer hertrainde
            # voor N uitslagen terwijl elke run na de eerste identiek was.
            record_result(match_id, g1, g2, retrain=False)
            matches_df = pd.read_csv(DATA_DIR / "matches.csv")  # reload
            new_results += 1

    if new_results == 0:
        print("  Geen nieuwe uitslagen gevonden")
    else:
        print(f"\n✓ {new_results} nieuwe uitslagen verwerkt")
        # FIX 5: hertraineer het model eenmalig na de volledige batch
        print("✓ ML-model hertrainen (eenmalig na batch)...")
        retrain_model()

# ── Dagelijkse status report ──────────────────────────────────────────────────

def print_status():
    matches_df = pd.read_csv(DATA_DIR / "matches.csv")
    total      = len(matches_df)
    played     = matches_df["played"].sum()
    remaining  = total - played
    tbds       = (matches_df["home"] == "TBD").sum() + (matches_df["away"] == "TBD").sum()

    print(f"\n── Status WK 2026 ───────────────────────────────────")
    print(f"  Gespeeld      : {int(played)}/{total}")
    print(f"  Resterend     : {remaining}")
    print(f"  TBD teams     : {tbds}")
    print(f"  Datum vandaag : {date.today()}")

    if played > 0:
        recent = matches_df[matches_df["played"] == 1].tail(3)
        print(f"\n  Laatste uitslagen:")
        for _, row in recent.iterrows():
            print(f"    {row['home']} {int(row['home_score'])}–{int(row['away_score'])} {row['away']}")

# ── Handmatige TBD update ─────────────────────────────────────────────────────

def manual_update_tbd(group: str, team_name: str):
    """
    Vul handmatig een TBD in voor een specifieke groep.
    Gebruik als openfootball nog niet bijgewerkt is.

    Voorbeeld: manual_update_tbd("B", "Italy")
    """
    matches_df = pd.read_csv(DATA_DIR / "matches.csv")
    mask = (matches_df["group"] == group)
    updated = 0

    for idx, row in matches_df[mask].iterrows():
        if row["home"] == "TBD":
            matches_df.at[idx, "home"] = team_name
            updated += 1
        if row["away"] == "TBD":
            matches_df.at[idx, "away"] = team_name
            updated += 1

    if updated > 0:
        matches_df.to_csv(DATA_DIR / "matches.csv", index=False)
        print(f"✓ Groep {group}: TBD → {team_name} ({updated} wedstrijden)")
    else:
        print(f"  Geen TBDs gevonden in groep {group}")

# ── Knockout fixture propagatie ───────────────────────────────────────────────
# Elke tuple: (stage, match_id_A, match_id_B, volgende_match_id, datum, tijd, use_losers)
# use_losers=True → troostfinale (verliezers spelen), anders winnaars.
_KO_PAIRS = [
    ("R16", 73,  74,  89,  "2026-07-06", "21:00", False),
    ("R16", 75,  76,  90,  "2026-07-07", "00:00", False),
    ("R16", 77,  78,  91,  "2026-07-07", "21:00", False),
    ("R16", 79,  80,  92,  "2026-07-08", "00:00", False),
    ("R16", 81,  82,  93,  "2026-07-08", "21:00", False),
    ("R16", 83,  84,  94,  "2026-07-09", "00:00", False),
    ("R16", 85,  86,  95,  "2026-07-09", "21:00", False),
    ("R16", 87,  88,  96,  "2026-07-10", "00:00", False),
    ("QF",  89,  90,  97,  "2026-07-12", "21:00", False),
    ("QF",  91,  92,  98,  "2026-07-12", "00:00", False),
    ("QF",  93,  94,  99,  "2026-07-13", "21:00", False),
    ("QF",  95,  96,  100, "2026-07-13", "00:00", False),
    ("SF",  97,  98,  101, "2026-07-16", "21:00", False),
    ("SF",  99,  100, 102, "2026-07-17", "21:00", False),
    ("3PL", 101, 102, 104, "2026-07-21", "21:00", True),   # troostfinale: verliezers
    ("F",   101, 102, 103, "2026-07-22", "21:00", False),  # finale: winnaars
]


def _ko_winner(row: pd.Series) -> str:
    """Geeft de winnaar van een gespeelde knockout-wedstrijd terug, of '' bij onduidelijkheid."""
    try:
        hs  = int(float(str(row["home_score"])))
        as_ = int(float(str(row["away_score"])))
    except (ValueError, TypeError):
        return ""
    if hs > as_:
        return str(row["home"])
    if as_ > hs:
        return str(row["away"])
    # Gelijkspel na verlengingen: check penalty_winner kolom
    pw = str(row.get("penalty_winner", "")).strip()
    return pw if pw and pw not in ("nan", "") else ""


def _ko_loser(row: pd.Series) -> str:
    """Geeft de verliezer van een gespeelde knockout-wedstrijd terug."""
    winner = _ko_winner(row)
    if not winner:
        return ""
    return str(row["away"]) if winner == str(row["home"]) else str(row["home"])


def propagate_knockout_fixtures() -> int:
    """
    Voeg de volgende knockout-ronde fixtures toe aan matches.csv zodra
    beide voorgaande wedstrijden in een bracket-paar gespeeld zijn.

    Werkt door alle ronden heen: R32→R16→QF→SF→Finale+Troostfinale.
    Veilig om meerdere keren te draaien: bestaande fixtures worden overgeslagen.
    Geeft het aantal nieuw toegevoegde fixtures terug.
    """
    matches_df = pd.read_csv(DATA_DIR / "matches.csv")
    added      = 0

    for stage, id_a, id_b, next_id, datum, tijd, use_losers in _KO_PAIRS:
        # Al ingepland?
        if (matches_df["match_id"] == next_id).any():
            continue

        row_a = matches_df[matches_df["match_id"] == id_a]
        row_b = matches_df[matches_df["match_id"] == id_b]

        if row_a.empty or row_b.empty:
            continue
        if int(row_a.iloc[0]["played"]) != 1 or int(row_b.iloc[0]["played"]) != 1:
            continue

        get_team = _ko_loser if use_losers else _ko_winner
        team_home = get_team(row_a.iloc[0])
        team_away = get_team(row_b.iloc[0])

        if not team_home or not team_away:
            print(f"  ⚠️  Winnaar van match {id_a} of {id_b} onduidelijk — voeg {stage} fixture {next_id} handmatig toe")
            continue

        new_row = pd.DataFrame([{
            "match_id":   next_id,
            "date":       datum,
            "time":       tijd,
            "group":      stage,
            "stage":      stage,
            "home":       team_home,
            "away":       team_away,
            "home_score": "",
            "away_score": "",
            "played":     0,
        }])
        matches_df = pd.concat([matches_df, new_row], ignore_index=True)
        print(f"  ✓ {stage} fixture toegevoegd (match {next_id}): {team_home} vs {team_away} — {datum}")
        added += 1

    if added > 0:
        matches_df.to_csv(DATA_DIR / "matches.csv", index=False)
        print(f"\n✓ {added} nieuwe knockout fixture(s) aan matches.csv toegevoegd")
    else:
        print("  Geen nieuwe fixtures te propageren")

    return added


# ── Hoofdfunctie ──────────────────────────────────────────────────────────────

def run_daily_update():
    print(f"\n{'='*50}")
    print(f"  WK 2026 AUTO-UPDATER — {date.today()}")
    print(f"{'='*50}\n")

    # 1. Data ophalen
    print("1. Openfootball data ophalen...")
    wc_data = fetch_wc2026()

    if wc_data:
        print(f"   ✓ {len(wc_data.get('matches', []))} wedstrijden gevonden")

        # 2. TBDs — alleen handmatig via update_tbds.py op 31 maart
        print("\n2. TBD-teams: gebruik 'python update_tbds.py' na 31 maart")
        print("   Of handmatig: python -m engine.auto_updater tbd B Italy")

        # 3. Nieuwe uitslagen verwerken
        print("\n3. Nieuwe uitslagen synchroniseren...")
        sync_results(wc_data)
    else:
        print("   ⚠️  Geen data beschikbaar — handmatig invullen via update_tbds.py")

    # 4. Volgende ronde fixtures propageren op basis van bekende winnaars
    print("\n4. Volgende ronde fixtures bepalen...")
    propagate_knockout_fixtures()

    # 5. Status
    print_status()
    print(f"\n{'='*50}\n")

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4 and sys.argv[1] == "tbd":
        group = sys.argv[2].upper()
        team  = sys.argv[3]
        manual_update_tbd(group, team)
    elif len(sys.argv) == 2 and sys.argv[1] == "propagate":
        propagate_knockout_fixtures()
    else:
        run_daily_update()