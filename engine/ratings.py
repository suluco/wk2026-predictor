import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Constanten ────────────────────────────────────────────────────────────────

# Hoe snel het model leert van nieuwe uitslagen (0 = niets, 1 = alles)
LEARNING_RATE = 0.08

# Hoe zwaar weegt een WK-wedstrijd vs historische data
WC_WEIGHT = 1.5

# ── Helpers ───────────────────────────────────────────────────────────────────

def expected_goals(attack: float, defense: float) -> float:
    return attack * defense

def update_team(team: pd.Series, scored: int, conceded: int, weight: float = 1.0) -> pd.Series:
    """
    Pas attack en defense van een team aan op basis van werkelijke uitslag.

    Logica:
      - Meer goals dan verwacht → attack omhoog, tegenstander defense omhoog
      - Minder goals dan verwacht → attack omlaag
      - Gewicht bepaalt hoe hard de update is (WK-wedstrijden wegen zwaarder)
    """
    team = team.copy()

    exp_scored    = team["attack"] * 1.05   # proxy: verwacht vs gemiddelde verdediging
    exp_conceded  = team["defense"] * 1.05

    # Attack update
    attack_delta  = (scored - exp_scored) / exp_scored
    team["attack"] = max(0.4, team["attack"] + LEARNING_RATE * weight * attack_delta * team["attack"])

    # Defense update (lager = beter)
    defense_delta = (conceded - exp_conceded) / exp_conceded
    team["defense"] = max(0.5, team["defense"] + LEARNING_RATE * weight * defense_delta * team["defense"])

    # Vorm update: gewogen gemiddelde van oude vorm + match resultaat
    if scored > conceded:
        match_form = 1.0    # gewonnen
    elif scored == conceded:
        match_form = 0.5    # gelijkspel
    else:
        match_form = 0.0    # verloren

    team["form"] = round(0.7 * team["form"] + 0.3 * match_form, 4)

    # Afronden
    team["attack"]  = round(team["attack"], 4)
    team["defense"] = round(team["defense"], 4)

    return team

# ── Hoofd update functie ──────────────────────────────────────────────────────

def apply_result(
    teams_df: pd.DataFrame,
    home: str,
    away: str,
    home_score: int,
    away_score: int
) -> pd.DataFrame:
    """
    Verwerk een gespeelde uitslag en update beide teams in de dataframe.
    Sla de bijgewerkte data op naar teams.csv.
    """
    teams_df = teams_df.copy()

    if home in teams_df.index:
        teams_df.loc[home] = update_team(
            teams_df.loc[home], home_score, away_score, weight=WC_WEIGHT
        )

    if away in teams_df.index:
        teams_df.loc[away] = update_team(
            teams_df.loc[away], away_score, home_score, weight=WC_WEIGHT
        )

    # Opslaan
    teams_df.to_csv(DATA_DIR / "teams.csv")
    print(f"✓ Update verwerkt: {home} {home_score}–{away_score} {away}")
    print(f"  {home}: attack={teams_df.loc[home]['attack']}, defense={teams_df.loc[home]['defense']}, form={teams_df.loc[home]['form']}")
    if away in teams_df.index:
        print(f"  {away}: attack={teams_df.loc[away]['attack']}, defense={teams_df.loc[away]['defense']}, form={teams_df.loc[away]['form']}")

    return teams_df

# ── Matches.csv updaten ───────────────────────────────────────────────────────

def record_result(match_id: int, home_score: int, away_score: int):
    """
    Schrijf de uitslag terug naar matches.csv en update teamsterktes.
    """
    from engine.simulator import load_teams

    matches_df = pd.read_csv(DATA_DIR / "matches.csv")
    teams_df   = load_teams()

    mask = matches_df["match_id"] == match_id
    if not mask.any():
        print(f"✗ match_id {match_id} niet gevonden")
        return

    row = matches_df[mask].iloc[0]
    home = row["home"]
    away = row["away"]

    # Update matches.csv
    matches_df.loc[mask, "home_score"] = home_score
    matches_df.loc[mask, "away_score"] = away_score
    matches_df.loc[mask, "played"]     = 1
    matches_df.to_csv(DATA_DIR / "matches.csv", index=False)

    # Update teamsterktes
    apply_result(teams_df, home, away, home_score, away_score)

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from engine.simulator import load_teams, simulate_match

    df = load_teams()

    print("VOOR update:")
    print(f"  France  → attack={df.loc['France','attack']}, defense={df.loc['France','defense']}, form={df.loc['France','form']}")
    print(f"  Senegal → attack={df.loc['Senegal','attack']}, defense={df.loc['Senegal','defense']}, form={df.loc['Senegal','form']}")

    # Simuleer: France wint met 3-0 van Senegal
    df_updated = apply_result(df, "France", "Senegal", 3, 0)

    print("\nNA update (France 3–0 Senegal):")
    print(f"  France  → attack={df_updated.loc['France','attack']}, defense={df_updated.loc['France','defense']}, form={df_updated.loc['France','form']}")
    print(f"  Senegal → attack={df_updated.loc['Senegal','attack']}, defense={df_updated.loc['Senegal','defense']}, form={df_updated.loc['Senegal','form']}")

    # Reset (zodat de test de CSV niet permanent aanpast)
    df.to_csv(DATA_DIR / "teams.csv")
    print("\n✓ teams.csv teruggezet naar origineel")