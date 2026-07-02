import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

LEARNING_RATE = 0.08
WC_WEIGHT     = 1.5

def update_team(team: pd.Series, scored: int, conceded: int, weight: float = 1.0) -> pd.Series:
    team = team.copy()

    exp_scored   = team["attack"] * 1.05
    exp_conceded = team["defense"] * 1.05

    attack_delta  = (scored - exp_scored) / exp_scored
    team["attack"] = max(0.4, team["attack"] + LEARNING_RATE * weight * attack_delta * team["attack"])

    defense_delta = (conceded - exp_conceded) / exp_conceded
    team["defense"] = max(0.5, team["defense"] + LEARNING_RATE * weight * defense_delta * team["defense"])

    if scored > conceded:
        match_form = 1.0
    elif scored == conceded:
        match_form = 0.5
    else:
        match_form = 0.0

    team["form"]    = round(0.7 * team["form"] + 0.3 * match_form, 4)
    team["attack"]  = round(team["attack"], 4)
    team["defense"] = round(team["defense"], 4)
    return team

def update_elo(home: str, away: str, home_score: int, away_score: int):
    """Update elo_ratings.csv na een gespeelde wedstrijd."""
    from engine.elo import load_elo, update_elo as elo_update, result_to_score, K_FACTOR

    elo = load_elo()

    elo_h = elo.get(home, 1500)
    elo_a = elo.get(away, 1500)
    score = result_to_score(home_score, away_score)

    new_h, new_a = elo_update(elo_h, elo_a, score, k=K_FACTOR)
    elo[home] = new_h
    elo[away] = new_a

    # Opslaan
    rows = [{"team": t, "elo": e} for t, e in sorted(elo.items(), key=lambda x: -x[1])]
    pd.DataFrame(rows).to_csv(DATA_DIR / "elo_ratings.csv", index=False)
    print(f"  Elo {home}: {elo_h} → {new_h}")
    print(f"  Elo {away}: {elo_a} → {new_a}")

def apply_result(teams_df, home, away, home_score, away_score):
    teams_df = teams_df.copy()

    if home in teams_df.index:
        teams_df.loc[home] = update_team(teams_df.loc[home], home_score, away_score, weight=WC_WEIGHT)
    if away in teams_df.index:
        teams_df.loc[away] = update_team(teams_df.loc[away], away_score, home_score, weight=WC_WEIGHT)

    teams_df.to_csv(DATA_DIR / "teams.csv")
    print(f"✓ Teamsterktes bijgewerkt: {home} {home_score}–{away_score} {away}")
    return teams_df

def retrain_model() -> None:
    """
    FIX 5: exporteerbare wrapper zodat auto_updater (en andere callers) het model
    eenmalig kunnen hertrainen na een batch record_result(retrain=False) aanroepen.
    """
    from engine.ml_model import retrain_model as _retrain
    _retrain()


def record_result(match_id: int, home_score: int, away_score: int,
                  retrain: bool = True, penalty_winner: str = ""):
    """
    Verwerk een gespeelde uitslag:
    1. Schrijf naar matches.csv (incl. optionele penalty_winner bij gelijkspel in KO)
    2. Update teamsterktes (attack/defense/form)
    3. Update Elo-ratings
    4. Hertraineer ML-model (alleen als retrain=True)
    """
    from engine.simulator import load_teams

    matches_df = pd.read_csv(DATA_DIR / "matches.csv")
    teams_df   = load_teams()

    mask = matches_df["match_id"] == match_id
    if not mask.any():
        print(f"✗ match_id {match_id} niet gevonden")
        return

    row  = matches_df[mask].iloc[0]
    home = row["home"]
    away = row["away"]

    # 1. matches.csv updaten
    if "penalty_winner" not in matches_df.columns:
        matches_df["penalty_winner"] = ""
    matches_df.loc[mask, "home_score"]     = home_score
    matches_df.loc[mask, "away_score"]     = away_score
    matches_df.loc[mask, "played"]         = 1
    if penalty_winner:
        matches_df.loc[mask, "penalty_winner"] = penalty_winner
    matches_df.to_csv(DATA_DIR / "matches.csv", index=False)
    print(f"\n✓ Uitslag opgeslagen: {home} {home_score}–{away_score} {away}")

    # 2. Teamsterktes updaten
    apply_result(teams_df, home, away, home_score, away_score)

    # 3. Elo updaten
    print("✓ Elo bijwerken...")
    update_elo(home, away, home_score, away_score)

    # 4. ML-model hertrainen op nieuwe data (overgeslagen bij retrain=False voor batches)
    if retrain:
        print("✓ ML-model hertrainen...")
        retrain_model()