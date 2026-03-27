import pandas as pd
import numpy as np
from pathlib import Path
from engine.elo import get_elo, elo_win_probability, load_elo
from engine.h2h import get_h2h_stats, load_h2h
from engine.simulator import load_teams, get_team
from engine.climate import get_climate_features, load_stadiums
from engine.fatigue import get_fatigue_diff
from engine.squad_strength import get_squad_features

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Feature kolommen ──────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Elo
    "elo_diff", "elo_win_prob_a", "elo_draw_prob",
    # Team stats
    "attack_diff", "defense_diff", "form_diff",
    "wc_exp_diff", "lambda_a", "lambda_b", "lambda_diff",
    # Confederatie
    "same_conf",
    # Squad
    "squad_score_diff", "squad_rating_diff", "ucl_players_diff",
    "star_player_diff", "coach_rating_diff", "age_factor_diff",
    # Klimaat
    "climate_adv_diff", "alt_penalty", "heat_penalty",
    # Vermoeidheid
    "days_rest_diff", "travel_km_diff", "jetlag_diff", "fatigue_diff",
]

# ── Feature vector bouwen ─────────────────────────────────────────────────────

def build_features(
    team_a: str,
    team_b: str,
    elo_dict: dict,
    h2h_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    match_id: int = 1,
    match_date: str = "2026-06-11",
    matches_df: pd.DataFrame = None,
    stadiums_df: pd.DataFrame = None,
) -> dict:
    """
    Bouw complete feature vector voor team_a vs team_b.
    Combineert: Elo + Team stats + Squad + Klimaat + Vermoeidheid + H2H
    """

    # ── Elo features ──────────────────────────────────────────────────
    elo_a = get_elo(team_a, elo_dict)
    elo_b = get_elo(team_b, elo_dict)
    elo_diff = elo_a - elo_b
    win_prob_a, draw_prob, win_prob_b = elo_win_probability(elo_a, elo_b)

    # ── Team stats ────────────────────────────────────────────────────
    ta = get_team(teams_df, team_a)
    tb = get_team(teams_df, team_b)

    attack_diff  = float(ta["attack"])  - float(tb["attack"])
    defense_diff = float(ta["defense"]) - float(tb["defense"])
    form_diff    = float(ta["form"])    - float(tb["form"])
    wc_exp_diff  = float(ta["wc_appearances"]) - float(tb["wc_appearances"])
    lambda_a     = float(ta["attack"]) * float(tb["defense"])
    lambda_b     = float(tb["attack"]) * float(ta["defense"])
    lambda_diff  = lambda_a - lambda_b
    same_conf    = int(ta["confederation"] == tb["confederation"])

    # ── Squad features ────────────────────────────────────────────────
    squad = get_squad_features(team_a, team_b)

    # ── Klimaat features ──────────────────────────────────────────────
    if stadiums_df is None:
        stadiums_df = load_stadiums()

    conf_a = ta["confederation"]
    conf_b = tb["confederation"]
    climate = get_climate_features(
        match_id, match_date, conf_a, conf_b, stadiums_df, use_api=False
    )

    # ── Vermoeidheid features ─────────────────────────────────────────
    if matches_df is None:
        matches_df = pd.read_csv(DATA_DIR / "matches.csv")

    fatigue = get_fatigue_diff(
        team_a, team_b, match_id, match_date, matches_df, stadiums_df
    )

    # ── H2H (bewaard als apart dict, niet in ML features) ────────────
    h2h = get_h2h_stats(team_a, team_b, h2h_df)

    return {
        # Elo
        "elo_a":            elo_a,
        "elo_b":            elo_b,
        "elo_diff":         elo_diff,
        "elo_win_prob_a":   win_prob_a,
        "elo_draw_prob":    draw_prob,
        # Team stats
        "attack_diff":      attack_diff,
        "defense_diff":     defense_diff,
        "form_diff":        form_diff,
        "wc_exp_diff":      wc_exp_diff,
        "lambda_a":         lambda_a,
        "lambda_b":         lambda_b,
        "lambda_diff":      lambda_diff,
        "same_conf":        same_conf,
        # Squad
        "squad_score_diff":  squad["squad_score_diff"],
        "squad_rating_diff": squad["squad_rating_diff"],
        "ucl_players_diff":  squad["ucl_players_diff"],
        "star_player_diff":  squad["star_player_diff"],
        "coach_rating_diff": squad["coach_rating_diff"],
        "age_factor_diff":   squad["age_factor_diff"],
        # Klimaat
        "climate_adv_diff":  climate["climate_adv_diff"],
        "alt_penalty":       climate["alt_penalty"],
        "heat_penalty":      climate["heat_penalty"],
        # Vermoeidheid
        "days_rest_diff":    fatigue["days_rest_diff"],
        "travel_km_diff":    fatigue["travel_km_diff"],
        "jetlag_diff":       fatigue["jetlag_diff"],
        "fatigue_diff":      fatigue["fatigue_diff"],
        # H2H apart (niet in ML)
        "h2h":              h2h,
    }

def features_to_array(features: dict) -> np.ndarray:
    return np.array([features[c] for c in FEATURE_COLS], dtype=np.float32)

# ── Trainingsdata ─────────────────────────────────────────────────────────────

def build_training_data() -> pd.DataFrame:
    import json

    elo_dict    = load_elo()
    h2h_df      = load_h2h()
    teams_df    = load_teams()
    stadiums_df = load_stadiums()

    rows = []

    for year in [2014, 2018, 2022]:
        path = DATA_DIR / f"wc{year}.json"
        if not path.exists():
            continue

        data     = json.loads(path.read_text())
        elo_run  = {}
        match_id = 1

        for match in data.get("matches", []):
            team_a = match.get("team1", "")
            team_b = match.get("team2", "")
            score  = match.get("score", {})
            date   = match.get("date", "2022-11-20")

            if not score:
                continue
            ft = score.get("ft", None)
            if not ft or len(ft) < 2:
                continue

            g_a, g_b = int(ft[0]), int(ft[1])

            if team_a not in elo_run:
                elo_run[team_a] = elo_dict.get(team_a, 1500)
            if team_b not in elo_run:
                elo_run[team_b] = elo_dict.get(team_b, 1500)

            try:
                feats = build_features(
                    team_a, team_b,
                    elo_run, h2h_df, teams_df,
                    match_id=match_id,
                    match_date=date,
                    stadiums_df=stadiums_df,
                )
            except Exception as e:
                match_id += 1
                continue

            label = 2 if g_a > g_b else (1 if g_a == g_b else 0)

            row = {k: v for k, v in feats.items() if k != "h2h"}
            row.update({
                "label": label, "year": year,
                "team_a": team_a, "team_b": team_b,
                "goals_a": g_a, "goals_b": g_b,
            })
            rows.append(row)
            match_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "training_data.csv", index=False)
    print(f"✓ training_data.csv opgeslagen ({len(df)} wedstrijden, {len(FEATURE_COLS)} features)")
    return df

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    elo_dict    = load_elo()
    h2h_df      = load_h2h()
    teams_df    = load_teams()
    stadiums_df = load_stadiums()
    matches_df  = pd.read_csv(DATA_DIR / "matches.csv")

    print("Feature vector — Netherlands vs Argentina (match 10):\n")
    feats = build_features(
        "Netherlands", "Argentina",
        elo_dict, h2h_df, teams_df,
        match_id=10, match_date="2026-06-14",
        matches_df=matches_df, stadiums_df=stadiums_df,
    )

    for k, v in feats.items():
        if k == "h2h":
            continue
        print(f"  {k:<22}: {round(v,4) if isinstance(v, float) else v}")

    print(f"\nArray shape: {features_to_array(feats).shape}")
    print(f"Features: {len(FEATURE_COLS)}")

    print("\nTrainingsdata bouwen...")
    df = build_training_data()
    print(f"\nLabel verdeling:")
    print(df["label"].value_counts().rename({0:"B wint", 1:"Gelijkspel", 2:"A wint"}))