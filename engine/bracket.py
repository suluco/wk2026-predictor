import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Groepsstand berekenen ─────────────────────────────────────────────────────

def compute_standings(group: str, matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Bereken stand voor een groep op basis van gespeelde wedstrijden.
    """
    group_matches = matches_df[
        (matches_df["group"] == group) &
        (matches_df["played"] == 1)
    ]

    teams = set()
    for _, row in group_matches.iterrows():
        teams.add(row["home"])
        teams.add(row["away"])

    stats = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "played": 0} for t in teams}

    for _, row in group_matches.iterrows():
        h, a = row["home"], row["away"]
        gh, ga = int(row["home_score"]), int(row["away_score"])

        stats[h]["gf"] += gh
        stats[h]["ga"] += ga
        stats[h]["gd"] += gh - ga
        stats[h]["played"] += 1

        stats[a]["gf"] += ga
        stats[a]["ga"] += gh
        stats[a]["gd"] += ga - gh
        stats[a]["played"] += 1

        if gh > ga:
            stats[h]["pts"] += 3
        elif gh == ga:
            stats[h]["pts"] += 1
            stats[a]["pts"] += 1
        else:
            stats[a]["pts"] += 3

    df = pd.DataFrame([
        {"team": t, **s} for t, s in stats.items()
    ]).sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)

    df["pos"] = range(1, len(df) + 1)
    return df

def get_group_winners(matches_df: pd.DataFrame) -> dict:
    """
    Geef groepswinnaar en runner-up per groep.
    Alleen voor groepen waar alle 6 wedstrijden gespeeld zijn.
    """
    results = {}
    for group in "ABCDEFGHIJKL":
        group_matches = matches_df[matches_df["group"] == group]
        played = group_matches[group_matches["played"] == 1]

        if len(played) < 6:
            continue

        standings = compute_standings(group, matches_df)
        if len(standings) >= 2:
            results[group] = {
                "winner":     standings.iloc[0]["team"],
                "runner_up":  standings.iloc[1]["team"],
                "third":      standings.iloc[2]["team"] if len(standings) > 2 else None,
            }

    return results

def predict_bracket(resources: dict = None) -> dict:
    """
    Simuleer de volledige knock-out bracket op basis van:
    - Bekende groepswinnaars (gespeelde wedstrijden)
    - Voorspelde groepswinnaars (niet gespeelde wedstrijden)
    """
    from engine.predictor import predict_match

    matches_df = pd.read_csv(DATA_DIR / "matches.csv")

    if resources is None:
        from engine.predictor import load_resources
        resources = load_resources()

    # Bepaal groepswinnaars
    winners = get_group_winners(matches_df)

    # Voor groepen zonder resultaten: voorspel groepswinnaar via model
    for group in "ABCDEFGHIJKL":
        if group not in winners:
            group_teams = matches_df[
                (matches_df["group"] == group) &
                (matches_df["home"] != "TBD") &
                (matches_df["away"] != "TBD")
            ][["home", "away"]].values.flatten()

            unique_teams = list(dict.fromkeys(group_teams))[:4]
            if len(unique_teams) >= 2:
                # Simpele aanname: sterkste team wint groep
                from engine.elo import get_elo
                elo_dict = resources["elo_dict"]
                ranked = sorted(unique_teams, key=lambda t: get_elo(t, elo_dict), reverse=True)
                winners[group] = {
                    "winner":    ranked[0],
                    "runner_up": ranked[1] if len(ranked) > 1 else "TBD",
                    "third":     ranked[2] if len(ranked) > 2 else "TBD",
                }

    def play(team_a: str, team_b: str) -> str:
        if team_a == "TBD" or team_b == "TBD":
            return team_a if team_b == "TBD" else team_b
        result = predict_match(team_a, team_b, knockout=True, n=25_000, resources=resources)
        return result["winner"]

    # WK 2026 Round of 32 koppelingen (vereenvoudigd schema)
    r32 = {}
    matchups = [
        ("A1","B2"),("C1","D2"),("E1","F2"),("G1","H2"),
        ("I1","J2"),("K1","L2"),("A2","B1"),("C2","D1"),
        ("E2","F1"),("G2","H1"),("I2","J1"),("K2","L1"),
        ("A3","B3"),("C3","D3"),("E3","F3"),("G3","H3"),
    ]

    def get_team(code: str) -> str:
        group = code[0]
        pos   = int(code[1])
        if group not in winners:
            return "TBD"
        if pos == 1:
            return winners[group]["winner"]
        elif pos == 2:
            return winners[group]["runner_up"]
        else:
            return winners[group].get("third", "TBD")

    print("\n── Round of 32 ──────────────────────────────")
    r32_winners = []
    for a_code, b_code in matchups:
        a = get_team(a_code)
        b = get_team(b_code)
        w = play(a, b)
        r32_winners.append(w)
        print(f"  {a:<20} vs {b:<20} → {w}")

    print("\n── Round of 16 ──────────────────────────────")
    r16_winners = []
    for i in range(0, len(r32_winners), 2):
        a = r32_winners[i]
        b = r32_winners[i+1] if i+1 < len(r32_winners) else "TBD"
        w = play(a, b)
        r16_winners.append(w)
        print(f"  {a:<20} vs {b:<20} → {w}")

    print("\n── Kwartfinales ─────────────────────────────")
    qf_winners = []
    for i in range(0, len(r16_winners), 2):
        a = r16_winners[i]
        b = r16_winners[i+1] if i+1 < len(r16_winners) else "TBD"
        w = play(a, b)
        qf_winners.append(w)
        print(f"  {a:<20} vs {b:<20} → {w}")

    print("\n── Halve finales ────────────────────────────")
    sf_winners = []
    sf_losers  = []
    for i in range(0, len(qf_winners), 2):
        a = qf_winners[i]
        b = qf_winners[i+1] if i+1 < len(qf_winners) else "TBD"
        result = predict_match(a, b, knockout=True, n=25_000, resources=resources)
        w = result["winner"]
        l = b if w == a else a
        sf_winners.append(w)
        sf_losers.append(l)
        print(f"  {a:<20} vs {b:<20} → {w}")

    # Finale + troostfinale
    finalist_a = sf_winners[0] if len(sf_winners) > 0 else "TBD"
    finalist_b = sf_winners[1] if len(sf_winners) > 1 else "TBD"
    third_a    = sf_losers[0]  if len(sf_losers) > 0  else "TBD"
    third_b    = sf_losers[1]  if len(sf_losers) > 1  else "TBD"

    champion      = play(finalist_a, finalist_b)
    third_place   = play(third_a, third_b)

    print(f"\n── Finale ───────────────────────────────────")
    print(f"  {finalist_a:<20} vs {finalist_b:<20} → 🏆 {champion}")
    print(f"\n── Troostfinale ─────────────────────────────")
    print(f"  {third_a:<20} vs {third_b:<20} → {third_place}")

    return {
        "r32":        r32_winners,
        "r16":        r16_winners,
        "qf":         qf_winners,
        "sf":         sf_winners,
        "finalist_a": finalist_a,
        "finalist_b": finalist_b,
        "champion":   champion,
        "third":      third_place,
    }

if __name__ == "__main__":
    from engine.predictor import load_resources
    resources = load_resources()
    result = predict_bracket(resources)
    print(f"\n🏆 Voorspelde wereldkampioen: {result['champion']}")
    print(f"🥉 Derde plaats: {result['third']}")