import pandas as pd
import numpy as np
from pathlib import Path
from engine.simulator import load_teams, simulate_match, predicted_winner
from engine.elo import load_elo, get_elo, elo_win_probability
from engine.h2h import load_h2h, get_h2h_stats
from engine.features import build_features, features_to_array
from engine.ml_model import load_model, predict_proba, blend_predictions

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Resources laden ───────────────────────────────────────────────────────────

def load_resources():
    return {
        "teams_df": load_teams(),
        "elo_dict": load_elo(),
        "h2h_df":   load_h2h(),
        "model":    load_model(),
    }

# ── Uitleg genereren ──────────────────────────────────────────────────────────

def explain(result: dict, feats: dict, h2h: dict, teams_df: pd.DataFrame) -> list[str]:
    reasons = []
    a, b = result["home"], result["away"]

    def get(team, col):
        if team in teams_df.index:
            return teams_df.loc[team, col]
        return None

    # Elo
    elo_a = feats.get("elo_a", 0)
    elo_b = feats.get("elo_b", 0)
    if elo_a > elo_b + 30:
        reasons.append(f"{a} heeft hogere Elo-rating ({int(elo_a)} vs {int(elo_b)})")
    elif elo_b > elo_a + 30:
        reasons.append(f"{b} heeft hogere Elo-rating ({int(elo_b)} vs {int(elo_a)})")
    else:
        reasons.append(f"Elo-ratings zijn nagenoeg gelijk ({int(elo_a)} vs {int(elo_b)})")

    # Aanval
    atk_diff = feats.get("attack_diff", 0)
    if atk_diff > 0.15:
        reasons.append(f"{a} heeft sterkere aanval (+{round(atk_diff,2)})")
    elif atk_diff < -0.15:
        reasons.append(f"{b} heeft sterkere aanval (+{round(-atk_diff,2)})")

    # Verdediging (lager defense-getal = beter)
    def_diff = feats.get("defense_diff", 0)
    if def_diff < -0.10:
        ta = teams_df.loc[a] if a in teams_df.index else None
        tb = teams_df.loc[b] if b in teams_df.index else None
        reasons.append(f"{a} heeft sterkere verdediging (def {round(float(ta['defense']),2) if ta is not None else '?'} vs {round(float(tb['defense']),2) if tb is not None else '?'})")
    elif def_diff > 0.10:
        ta = teams_df.loc[a] if a in teams_df.index else None
        tb = teams_df.loc[b] if b in teams_df.index else None
        reasons.append(f"{b} heeft sterkere verdediging (def {round(float(tb['defense']),2) if tb is not None else '?'} vs {round(float(ta['defense']),2) if ta is not None else '?'})")

    # Vorm
    form_diff = feats.get("form_diff", 0)
    if form_diff > 0.10:
        reasons.append(f"{a} is in betere vorm")
    elif form_diff < -0.10:
        reasons.append(f"{b} is in betere vorm")

    # WK-ervaring
    exp_diff = feats.get("wc_exp_diff", 0)
    if exp_diff > 4:
        reasons.append(f"{a} heeft significant meer WK-ervaring")
    elif exp_diff < -4:
        reasons.append(f"{b} heeft significant meer WK-ervaring")

    # H2H
    games = h2h.get("games", 0)
    if games > 0:
        reasons.append(
            f"H2H op WK ({games} duel{'s' if games>1 else ''}): "
            f"{a} {h2h['wins_a']}W – {h2h['draws']}G – {h2h['wins_b']}W {b}"
        )

    # Verwachte goals
    reasons.append(
        f"Model verwacht {result['exp_goals_home']} goals voor {a} "
        f"en {result['exp_goals_away']} voor {b}"
    )

    return reasons

# ── Hoofd voorspelling ────────────────────────────────────────────────────────

def predict_match(
    home: str,
    away: str,
    knockout: bool = False,
    n: int = 50_000,
    resources: dict = None,
    match_id: int = 1,
    match_date: str = "2026-06-11",
) -> dict:
    """
    Volledige voorspelling via drie lagen:
      1. Poisson Monte Carlo simulatie
      2. XGBoost ML-model
      3. H2H correctie
    """
    if resources is None:
        resources = load_resources()

    teams_df = resources["teams_df"]
    elo_dict = resources["elo_dict"]
    h2h_df   = resources["h2h_df"]
    model, scaler = resources["model"]

    # ── Laag 1: Poisson simulatie ─────────────────────────────────────
    sim = simulate_match(home, away, n=n, knockout=knockout, teams_df=teams_df)
    poisson_probs = {
        "win_a": sim["win_home"] / 100,
        "draw":  sim["draw"] / 100,
        "win_b": sim["win_away"] / 100,
    }

    # ── Laag 2: ML voorspelling ───────────────────────────────────────
    feats = build_features(
        home, away, elo_dict, h2h_df, teams_df,
        match_id=match_id, match_date=match_date,
    )
    arr = features_to_array(feats)
    ml_probs = predict_proba(arr, model, scaler)

    # ── Laag 3: Elo-kansen als derde blendingscomponent ──────────────
    elo_a = feats.get("elo_a", 0)
    elo_b = feats.get("elo_b", 0)
    elo_win_a, elo_draw, elo_win_b = elo_win_probability(elo_a, elo_b)
    elo_probs = {"win_a": elo_win_a, "draw": elo_draw, "win_b": elo_win_b}

    # ── Laag 4: Blend + H2H correctie ────────────────────────────────
    h2h = get_h2h_stats(home, away, h2h_df)
    blended = blend_predictions(ml_probs, poisson_probs, h2h, elo_probs=elo_probs)

    # ── Knockout aanpassing ───────────────────────────────────────────
    if knockout and blended["draw"] > 0:
        half = blended["draw"] / 2
        blended["win_a"] += half
        blended["win_b"] += half
        blended["draw"]   = 0.0

    # ── Zekerheid ────────────────────────────────────────────────────
    confidence = round(max(blended["win_a"], blended["draw"], blended["win_b"]) * 100, 1)

    # ── Winner bepalen ────────────────────────────────────────────────
    if blended["win_a"] >= blended["win_b"] and blended["win_a"] >= blended["draw"]:
        winner = home
    elif blended["win_b"] >= blended["win_a"] and blended["win_b"] >= blended["draw"]:
        winner = away
    else:
        winner = "Gelijkspel"

    result = {
        "home":             home,
        "away":             away,
        "win_home":         round(blended["win_a"] * 100, 1),
        "draw":             round(blended["draw"] * 100, 1),
        "win_away":         round(blended["win_b"] * 100, 1),
        "most_likely_score": sim["most_likely_score"],
        "top_scores":       sim["top_scores"],
        "exp_goals_home":   sim["exp_goals_home"],
        "exp_goals_away":   sim["exp_goals_away"],
        "confidence":       confidence,
        "winner":           winner,
        "knockout":         knockout,
        "ml_probs":         ml_probs,
        "poisson_probs":    poisson_probs,
        "elo_probs":        elo_probs,
    }

    result["reasons"] = explain(result, feats, h2h, teams_df)
    return result


def predict_all_upcoming(n: int = 25_000) -> pd.DataFrame:
    matches  = pd.read_csv(DATA_DIR / "matches.csv")
    upcoming = matches[(matches["played"] == 0)].copy()
    resources = load_resources()
    rows = []

    for _, row in upcoming.iterrows():
        home = row["home"]
        away = row["away"]
        if home == "TBD" or away == "TBD":
            continue
        result = predict_match(
            home, away, n=n, resources=resources,
            match_id=int(row["match_id"]),
            match_date=str(row["date"]),
        )
        rows.append({
            "match_id":   row["match_id"],
            "date":       row["date"],
            "group":      row["group"],
            "home":       home,
            "away":       away,
            "pred_home":  result["most_likely_score"][0],
            "pred_away":  result["most_likely_score"][1],
            "win_home":   result["win_home"],
            "draw":       result["draw"],
            "win_away":   result["win_away"],
            "confidence": result["confidence"],
            "winner":     result["winner"],
        })

    return pd.DataFrame(rows)


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    resources = load_resources()

    tests = [
        ("Netherlands", "England"),
        ("France",      "Brazil"),
        ("Argentina",   "Germany"),
    ]

    for home, away in tests:
        r = predict_match(home, away, resources=resources)
        print(f"\n{'='*48}")
        print(f"  {r['home']} vs {r['away']}")
        print(f"{'='*48}")
        print(f"  Score        : {r['most_likely_score'][0]}–{r['most_likely_score'][1]}")
        print(f"  Winnaar      : {r['winner']}")
        print(f"  Zekerheid    : {r['confidence']}%")
        print(f"  Win {r['home']:<14}: {r['win_home']}%")
        print(f"  Gelijkspel   : {r['draw']}%")
        print(f"  Win {r['away']:<14}: {r['win_away']}%")
        print(f"  [ML]     {r['ml_probs']}")
        print(f"  [Poisson]{r['poisson_probs']}")
        print(f"\n  Waarom:")
        for reason in r["reasons"]:
            print(f"    • {reason}")