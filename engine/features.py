import pandas as pd
import numpy as np
from pathlib import Path
from engine.elo import get_elo, elo_win_probability, load_elo
from engine.h2h import get_h2h_stats, load_h2h
from engine.simulator import load_teams, get_team

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Feature vector bouwen ─────────────────────────────────────────────────────

def build_features(
    team_a: str,
    team_b: str,
    elo_dict: dict,
    h2h_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> dict:
    """
    Bouw een complete feature vector voor team_a vs team_b.

    Features:
      Elo-gebaseerd      : elo_diff, elo_win_prob_a, elo_draw_prob
      Team stats         : attack_diff, defense_diff, form_diff
      H2H                : h2h_games, h2h_win_ratio_a, h2h_goal_diff
      Ervaring           : wc_exp_diff
      Confederatie       : zelfde confederatie (bool)
    """
    # ── Elo features ─────────────────────────────────────────────────
    elo_a = get_elo(team_a, elo_dict)
    elo_b = get_elo(team_b, elo_dict)
    elo_diff = elo_a - elo_b
    win_prob_a, draw_prob, win_prob_b = elo_win_probability(elo_a, elo_b)

    # ── Team stats features ───────────────────────────────────────────
    ta = get_team(teams_df, team_a)
    tb = get_team(teams_df, team_b)

    attack_diff  = float(ta["attack"])  - float(tb["attack"])
    defense_diff = float(ta["defense"]) - float(tb["defense"])
    form_diff    = float(ta["form"])    - float(tb["form"])
    wc_exp_diff  = float(ta["wc_appearances"]) - float(tb["wc_appearances"])

    # Verwachte goals (Poisson lambdas)
    lambda_a = float(ta["attack"]) * float(tb["defense"])
    lambda_b = float(tb["attack"]) * float(ta["defense"])
    lambda_diff = lambda_a - lambda_b

    # ── H2H features ─────────────────────────────────────────────────
    h2h = get_h2h_stats(team_a, team_b, h2h_df)
    h2h_games     = min(h2h["games"], 3)
    h2h_win_ratio = h2h["win_ratio_a"]
    h2h_goal_diff = np.clip(
        (h2h["goals_a"] - h2h["goals_b"]) / max(h2h["games"], 1),
    )

    # ── Confederatie feature ──────────────────────────────────────────
    same_conf = int(ta["confederation"] == tb["confederation"])

    return {
        # Elo
        "elo_a":          elo_a,
        "elo_b":          elo_b,
        "elo_diff":       elo_diff,
        "elo_win_prob_a": win_prob_a,
        "elo_draw_prob":  draw_prob,
        # Team stats
        "attack_diff":    attack_diff,
        "defense_diff":   defense_diff,
        "form_diff":      form_diff,
        "wc_exp_diff":    wc_exp_diff,
        "lambda_a":       lambda_a,
        "lambda_b":       lambda_b,
        "lambda_diff":    lambda_diff,
        # H2H
        "h2h_games":      h2h_games,
        "h2h_win_ratio":  h2h_win_ratio,
        "h2h_goal_diff":  h2h_goal_diff,
        # Conf
        "same_conf":      same_conf,
    }

def features_to_array(features: dict) -> np.ndarray:
    """Zet feature dict om naar numpy array voor ML-model."""
    cols = [
        "elo_diff", "elo_win_prob_a", "elo_draw_prob",
        "attack_diff", "defense_diff", "form_diff",
        "wc_exp_diff", "lambda_a", "lambda_b", "lambda_diff",
        "same_conf",
    ]
    return np.array([features[c] for c in cols], dtype=np.float32)

# ── Trainingsdata bouwen vanuit historische WK-wedstrijden ────────────────────

def build_training_data() -> pd.DataFrame:
    """
    Bouw trainingsdata op vanuit WK 2014, 2018, 2022.
    Elke wedstrijd = één rij met features + label (0=B wint, 1=gelijkspel, 2=A wint)
    """
    import json

    elo_dict = load_elo()
    h2h_df   = load_h2h()
    teams_df = load_teams()

    rows = []

    for year in [2014, 2018, 2022]:
        path = DATA_DIR / f"wc{year}.json"
        if not path.exists():
            continue

        data = json.loads(path.read_text())

        # Bouw Elo stap voor stap op (simuleer chronologische volgorde)
        elo_running = {}

        for match in data.get("matches", []):
            team_a = match.get("team1", "")
            team_b = match.get("team2", "")
            score  = match.get("score", {})

            if not score:
                continue
            ft = score.get("ft", None)
            if not ft or len(ft) < 2:
                continue

            g_a, g_b = int(ft[0]), int(ft[1])

            # Initialiseer Elo als nog niet gezien dit toernooi
            if team_a not in elo_running:
                elo_running[team_a] = elo_dict.get(team_a, 1500)
            if team_b not in elo_running:
                elo_running[team_b] = elo_dict.get(team_b, 1500)

            # Features op moment VAN de wedstrijd (voor de update)
            try:
                feats = build_features(team_a, team_b, elo_running, h2h_df, teams_df)
            except Exception:
                continue

            # Label: 0 = B wint, 1 = gelijkspel, 2 = A wint
            if g_a > g_b:
                label = 2
            elif g_a == g_b:
                label = 1
            else:
                label = 0

            row = feats.copy()
            row["label"]  = label
            row["year"]   = year
            row["team_a"] = team_a
            row["team_b"] = team_b
            row["goals_a"] = g_a
            row["goals_b"] = g_b
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "training_data.csv", index=False)
    print(f"✓ training_data.csv opgeslagen ({len(df)} wedstrijden)")
    return df

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Features bouwen voor Nederland vs Argentinië...")
    elo_dict = load_elo()
    h2h_df   = load_h2h()
    teams_df = load_teams()

    feats = build_features("Netherlands", "Argentina", elo_dict, h2h_df, teams_df)
    print("\nFeature vector:")
    for k, v in feats.items():
        print(f"  {k:<20}: {round(v, 4) if isinstance(v, float) else v}")

    arr = features_to_array(feats)
    print(f"\nArray shape: {arr.shape}")
    print(f"Array: {arr}")

    print("\nTrainingsdata bouwen...")
    df = build_training_data()
    print(f"\nLabel verdeling:")
    print(df["label"].value_counts().rename({0: "B wint", 1: "Gelijkspel", 2: "A wint"}))