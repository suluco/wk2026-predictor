import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

# ── Data laden ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"

def load_teams() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "teams.csv")
    df.set_index("team", inplace=True)
    return df

def get_team(df: pd.DataFrame, team: str) -> pd.Series:
    if team not in df.index:
        # TBD of onbekend team: gebruik gemiddelde waarden
        return pd.Series({
            "fifa_rank": 40,
            "attack": 1.00,
            "defense": 1.10,
            "form": 0.55,
            "wc_appearances": 2,
            "avg_goals_scored": 0.85,
            "avg_goals_conceded": 1.10,
            "confederation": "UNK"
        })
    return df.loc[team]

# ── Lambda berekening ─────────────────────────────────────────────────────────

def compute_lambda(team: pd.Series, opponent: pd.Series) -> float:
    """
    Bereken verwachte doelpunten voor 'team' tegen 'opponent'.

    Factoren:
      - Aanvalskracht van team
      - Verdedigingszwakte van tegenstander
      - Recente vorm van team (wegingsfactor)
      - WK-ervaring als kleine correctie
    """
    base = team["attack"] * opponent["defense"]

    # Vormcorrectie: form loopt van 0-1, we schalen naar 0.85-1.15
    form_factor = 0.85 + (team["form"] * 0.30)

    # Ervaringscorrectie: meer WK-ervaring = lichte bonus (max +3%)
    exp_factor = 1.0 + min(team["wc_appearances"] / 100, 0.03)

    return base * form_factor * exp_factor

# ── Monte Carlo simulatie ─────────────────────────────────────────────────────

def simulate_match(
    home: str,
    away: str,
    n: int = 50_000,
    knockout: bool = False,
    teams_df: pd.DataFrame = None
) -> dict:
    """
    Simuleer een wedstrijd tussen home en away via Monte Carlo + Poisson.

    Returns dict met:
      - win_home, draw, win_away    (kansen in %)
      - most_likely_score           (tuple)
      - top_scores                  (lijst van (score_str, pct))
      - exp_goals_home/away         (verwachte doelpunten)
      - confidence                  (zekerheidspercentage)
    """
    if teams_df is None:
        teams_df = load_teams()

    home_data = get_team(teams_df, home)
    away_data = get_team(teams_df, away)

    lam_home = compute_lambda(home_data, away_data)
    lam_away = compute_lambda(away_data, home_data)

    # Trek doelpunten uit Poisson-verdeling
    goals_home = np.random.poisson(lam_home, n)
    goals_away = np.random.poisson(lam_away, n)

    # Basisresultaten
    win_home = np.mean(goals_home > goals_away)
    draw     = np.mean(goals_home == goals_away)
    win_away = np.mean(goals_home < goals_away)

    # Meest voorkomende uitslagen
    score_counts = Counter(zip(goals_home, goals_away))
    top_raw = score_counts.most_common(8)
    most_likely_score = top_raw[0][0]
    top_scores = [(f"{s[0]}–{s[1]}", round(c / n * 100, 1)) for s, c in top_raw]

    # Zekerheidspercentage = kans op de meest waarschijnlijke uitkomst (win/draw/win)
    max_outcome_pct = round(max(win_home, draw, win_away) * 100, 1)

    # Knockout: bij gelijkspel gaat het naar verlengingen → 50/50 penalty simulatie
    if knockout and draw > 0:
        penalty_home = draw / 2
        penalty_away = draw / 2
        win_home += penalty_home
        win_away += penalty_away
        draw = 0.0

    return {
        "home": home,
        "away": away,
        "win_home": round(win_home * 100, 1),
        "draw":     round(draw * 100, 1),
        "win_away": round(win_away * 100, 1),
        "most_likely_score": most_likely_score,
        "top_scores": top_scores,
        "exp_goals_home": round(lam_home, 2),
        "exp_goals_away": round(lam_away, 2),
        "confidence": max_outcome_pct,
        "knockout": knockout,
    }

def predicted_winner(result: dict) -> str:
    if result["win_home"] >= result["win_away"] and result["win_home"] >= result["draw"]:
        return result["home"]
    if result["win_away"] >= result["win_home"] and result["win_away"] >= result["draw"]:
        return result["away"]
    return "Gelijkspel"

# ── Snelle test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_teams()
    result = simulate_match("Netherlands", "Argentina", n=50_000, teams_df=df)

    print(f"\n{'='*40}")
    print(f"  {result['home']} vs {result['away']}")
    print(f"{'='*40}")
    print(f"  Verwachte goals : {result['exp_goals_home']} – {result['exp_goals_away']}")
    print(f"  Meest likely    : {result['most_likely_score'][0]}–{result['most_likely_score'][1]}")
    print(f"  Win {result['home']:<12}: {result['win_home']}%")
    print(f"  Gelijkspel      : {result['draw']}%")
    print(f"  Win {result['away']:<12}: {result['win_away']}%")
    print(f"  Zekerheid       : {result['confidence']}%")
    print(f"  Voorspelling    : {predicted_winner(result)}")
    print(f"{'='*40}\n")
    print("Top uitslagen:")
    for score, pct in result['top_scores']:
        bar = "█" * int(pct)
        print(f"  {score:>5}  {bar} {pct}%")