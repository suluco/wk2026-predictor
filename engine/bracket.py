import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Groepsstand berekenen ─────────────────────────────────────────────────────

def compute_standings(group: str, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Bereken stand voor een groep op basis van gespeelde wedstrijden."""
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
    Geef groepswinnaar, runner-up en derde per groep.
    Alleen voor groepen waar alle 6 wedstrijden gespeeld zijn.
    """
    results = {}
    for group in "ABCDEFGHIJKL":
        group_matches = matches_df[matches_df["group"] == group]
        played = group_matches[group_matches["played"] == 1]

        if len(played) < 6:
            continue

        standings = compute_standings(group, matches_df)
        if len(standings) >= 3:
            results[group] = {
                "winner":    standings.iloc[0]["team"],
                "runner_up": standings.iloc[1]["team"],
                "third":     standings.iloc[2]["team"],
                "third_pts": int(standings.iloc[2]["pts"]),
                "third_gd":  int(standings.iloc[2]["gd"]),
                "third_gf":  int(standings.iloc[2]["gf"]),
            }

    return results


def _predict_group_standings(group: str, matches_df: pd.DataFrame, resources: dict) -> tuple:
    """
    Simuleer groepswedstrijden via het model en bereken de verwachte eindstand.
    Geeft een tuple terug: (ranked_list, expected_pts_dict, expected_gd_dict).
    FIX 2: returntype uitgebreid van list naar tuple zodat pts/gd beschikbaar zijn
    voor de derde-plaatsbepaling zonder terug te vallen op hardcoded waarden.
    FIX 4: except-blok logt nu de fout en gebruikt een neutrale 1-punt fallback
    (equivalente gelijkspelsverwachting) in plaats van Elo/1000 dat een andere
    schaal heeft dan echte wedstrijdpunten.
    """
    from engine.elo import get_elo

    group_teams = matches_df[
        (matches_df["group"] == group) &
        (matches_df["home"] != "TBD") &
        (matches_df["away"] != "TBD")
    ][["home", "away"]].values.flatten()

    unique_teams = list(dict.fromkeys(group_teams))[:4]
    elo_dict = resources["elo_dict"]

    # Simuleer elke match in de groep om verwachte punten te berekenen
    from engine.predictor import predict_match
    expected_pts = {t: 0.0 for t in unique_teams}
    expected_gd  = {t: 0.0 for t in unique_teams}

    for i, ta in enumerate(unique_teams):
        for tb in unique_teams[i+1:]:
            try:
                r = predict_match(ta, tb, knockout=False, n=10_000, resources=resources)
                # Verwachte punten op basis van winkansen
                expected_pts[ta] += r["win_home"] / 100 * 3 + r["draw"] / 100 * 1
                expected_pts[tb] += r["win_away"] / 100 * 3 + r["draw"] / 100 * 1
                # Verwacht doelsaldo
                gd = r["exp_goals_home"] - r["exp_goals_away"]
                expected_gd[ta] += gd
                expected_gd[tb] -= gd
            except Exception as e:
                # FIX 4: was 'get_elo(team) / 1000' — verkeerde schaal (~1.5–2.0) t.o.v.
                # echte wedstrijdpunten (~0.5–2.5). Vervangen door neutrale 1-punt fallback
                # (overeenkomend met een verwacht gelijkspel).
                print(f"  [bracket] Fallback voor {ta} vs {tb} in groep {group}: {e}")
                expected_pts[ta] += 1.0
                expected_pts[tb] += 1.0

    ranked = sorted(unique_teams,
                    key=lambda t: (expected_pts[t], expected_gd[t]),
                    reverse=True)
    return ranked, expected_pts, expected_gd


def predict_bracket(resources: dict = None) -> dict:
    """
    Simuleer de volledige knock-out bracket:
    - Groepswinnaars uit gespeelde matches
    - Niet-gespeelde groepen: verwachte stand via model-simulatie
    - R32: 12 winners vs runners-up + 4 matchen met beste 8 derden
    - R16 → QF → SF → Finale
    """
    from engine.predictor import predict_match

    matches_df = pd.read_csv(DATA_DIR / "matches.csv")

    if resources is None:
        from engine.predictor import load_resources
        resources = load_resources()

    # ── Stap 1: Bepaal groepsstand per groep ──────────────────────────────────
    winners = get_group_winners(matches_df)

    for group in "ABCDEFGHIJKL":
        if group not in winners:
            # FIX 2: unpack tuple — was 'ranked = ...' (list only); nu ook pts/gd beschikbaar
            ranked, exp_pts, exp_gd = _predict_group_standings(group, matches_df, resources)
            if len(ranked) >= 3:
                # FIX 2: vervang hardcoded (4, 0, 3) door werkelijk gesimuleerde waarden
                # zodat de ranking van beste derden teamsterkte weerspiegelt.
                winners[group] = {
                    "winner":    ranked[0],
                    "runner_up": ranked[1],
                    "third":     ranked[2],
                    "third_pts": exp_pts[ranked[2]],
                    "third_gd":  exp_gd[ranked[2]],
                    "third_gf":  0,  # goals-for niet getrackt in simulatie; neutrale waarde
                }
            elif len(ranked) == 2:
                winners[group] = {
                    "winner":    ranked[0],
                    "runner_up": ranked[1],
                    "third":     "TBD",
                    "third_pts": 0,
                    "third_gd":  0,
                    "third_gf":  0,
                }

    # ── Stap 2: Selecteer beste 8 derde plaatsen (uit alle 12 groepen) ─────────
    thirds = []
    for group, data in winners.items():
        if data["third"] and data["third"] != "TBD":
            thirds.append({
                "team":  data["third"],
                "group": group,
                "pts":   data["third_pts"],
                "gd":    data["third_gd"],
                "gf":    data["third_gf"],
            })

    thirds_sorted = sorted(thirds,
                           key=lambda x: (x["pts"], x["gd"], x["gf"]),
                           reverse=True)
    best8_thirds_info = thirds_sorted[:8]
    # Vul aan met TBD als minder dan 8 thirds bekend zijn
    while len(best8_thirds_info) < 8:
        best8_thirds_info.append({"team": "TBD", "group": "?", "pts": 0, "gd": 0, "gf": 0})

    def play(team_a: str, team_b: str) -> dict:
        if team_a == "TBD":
            return {"team_a": team_a, "team_b": team_b, "winner": team_b, "score": (0, 0), "win_a": 0, "win_b": 100, "draw": 0}
        if team_b == "TBD":
            return {"team_a": team_a, "team_b": team_b, "winner": team_a, "score": (0, 0), "win_a": 100, "win_b": 0, "draw": 0}

        # Gebruik de werkelijke uitslag als de wedstrijd al gespeeld is
        played_row = matches_df[
            (matches_df["played"] == 1) &
            (matches_df["stage"] != "group") &
            (
                ((matches_df["home"] == team_a) & (matches_df["away"] == team_b)) |
                ((matches_df["home"] == team_b) & (matches_df["away"] == team_a))
            )
        ]
        if not played_row.empty:
            row = played_row.iloc[0]
            hs  = int(row["home_score"])
            as_ = int(row["away_score"])
            a_score = hs if row["home"] == team_a else as_
            b_score = as_ if row["home"] == team_a else hs
            if a_score != b_score:
                winner = team_a if a_score > b_score else team_b
            else:
                # Verlengingen gelijkspel: check penalty_winner kolom
                pw = str(row.get("penalty_winner", "")).strip()
                winner = pw if pw and pw in (team_a, team_b) else team_a
            return {
                "team_a": team_a, "team_b": team_b, "winner": winner,
                "score": (a_score, b_score),
                "win_a": 100 if winner == team_a else 0,
                "win_b": 100 if winner == team_b else 0,
                "draw": 0, "actual": True,
            }

        r = predict_match(team_a, team_b, knockout=True, n=25_000, resources=resources)
        return {
            "team_a": team_a,
            "team_b": team_b,
            "winner": r["winner"],
            "score": r["most_likely_score"],
            "win_a": r["win_home"],
            "win_b": r["win_away"],
            "draw": r["draw"],
            "exp_home": r["exp_goals_home"],
            "exp_away": r["exp_goals_away"],
        }

    def g(group: str, pos: str) -> str:
        if group not in winners:
            return "TBD"
        return winners[group].get({"1": "winner", "2": "runner_up", "3": "third"}.get(pos, "winner"), "TBD") or "TBD"

    # ── Stap 3: Round of 32 ────────────────────────────────────────────────────
    # Gebruik officiële fixtures uit matches.csv als die er staan (stage == "R32"),
    # anders fallback naar berekening op basis van groepsstand.
    r32_fixture_rows = matches_df[matches_df["stage"] == "R32"]

    if len(r32_fixture_rows) >= 16:
        all_r32_matchups = [
            (row["home"], row["away"])
            for _, row in r32_fixture_rows.sort_values("match_id").iterrows()
        ]
    else:
        standard_matchups = [
            (g("A","1"), g("B","2")),
            (g("C","1"), g("D","2")),
            (g("E","1"), g("F","2")),
            (g("G","1"), g("H","2")),
            (g("I","1"), g("J","2")),
            (g("K","1"), g("L","2")),
            (g("B","1"), g("A","2")),
            (g("D","1"), g("C","2")),
            (g("F","1"), g("E","2")),
            (g("H","1"), g("G","2")),
            (g("J","1"), g("I","2")),
            (g("L","1"), g("K","2")),
        ]
        thirds_by_group = sorted(best8_thirds_info, key=lambda t: t["group"])
        thirds_ordered  = [t["team"] for t in thirds_by_group]
        while len(thirds_ordered) < 8:
            thirds_ordered.append("TBD")
        thirds_matchups = [
            (thirds_ordered[0], thirds_ordered[7]),
            (thirds_ordered[1], thirds_ordered[6]),
            (thirds_ordered[2], thirds_ordered[5]),
            (thirds_ordered[3], thirds_ordered[4]),
        ]
        all_r32_matchups = standard_matchups + thirds_matchups

    print("\n── Round of 32 ──────────────────────────────")
    r32_matchups = []
    r32_winners  = []
    for a, b in all_r32_matchups:
        m = play(a, b)
        r32_matchups.append(m)
        r32_winners.append(m["winner"])
        print(f"  {a:<22} vs {b:<22} → {m['score'][0]}-{m['score'][1]} → {m['winner']}")

    # ── Stap 4: Round of 16 ───────────────────────────────────────────────────
    # Gebruik officiële fixtures uit matches.csv als die er staan (stage == "R16"),
    # anders pair sequentieel de R32-winnaars (fallback).
    r16_fixture_rows = matches_df[matches_df["stage"] == "R16"]

    if len(r16_fixture_rows) >= 8:
        all_r16_matchups = [
            (row["home"], row["away"])
            for _, row in r16_fixture_rows.sort_values("match_id").iterrows()
        ]
    else:
        all_r16_matchups = [
            (r32_winners[i], r32_winners[i+1] if i+1 < len(r32_winners) else "TBD")
            for i in range(0, len(r32_winners), 2)
        ]

    print("\n── Round of 16 ──────────────────────────────")
    r16_matchups = []
    r16_winners  = []
    for a, b in all_r16_matchups:
        m = play(a, b)
        r16_matchups.append(m)
        r16_winners.append(m["winner"])
        print(f"  {a:<22} vs {b:<22} → {m['score'][0]}-{m['score'][1]} → {m['winner']}")

    # ── Stap 5: Kwartfinales ───────────────────────────────────────────────────
    # Gebruik officiële fixtures uit matches.csv als die er staan (stage == "QF"),
    # anders pair sequentieel de R16-winnaars (fallback).
    qf_fixture_rows = matches_df[matches_df["stage"] == "QF"]

    if len(qf_fixture_rows) >= 4:
        all_qf_matchups = [
            (row["home"], row["away"])
            for _, row in qf_fixture_rows.sort_values("match_id").iterrows()
        ]
    else:
        all_qf_matchups = [
            (r16_winners[i], r16_winners[i+1] if i+1 < len(r16_winners) else "TBD")
            for i in range(0, len(r16_winners), 2)
        ]

    print("\n── Kwartfinales ─────────────────────────────")
    qf_matchups = []
    qf_winners  = []
    for a, b in all_qf_matchups:
        m = play(a, b)
        qf_matchups.append(m)
        qf_winners.append(m["winner"])
        print(f"  {a:<22} vs {b:<22} → {m['score'][0]}-{m['score'][1]} → {m['winner']}")

    print("\n── Halve finales ────────────────────────────")
    sf_matchups = []
    sf_winners  = []
    sf_losers   = []
    for i in range(0, len(qf_winners), 2):
        a = qf_winners[i]
        b = qf_winners[i+1] if i+1 < len(qf_winners) else "TBD"
        m = play(a, b)
        sf_matchups.append(m)
        w = m["winner"]
        l = b if w == a else a
        sf_winners.append(w)
        sf_losers.append(l)
        print(f"  {a:<22} vs {b:<22} → {m['score'][0]}-{m['score'][1]} → {w}")

    finalist_a = sf_winners[0] if len(sf_winners) > 0 else "TBD"
    finalist_b = sf_winners[1] if len(sf_winners) > 1 else "TBD"
    third_a    = sf_losers[0]  if len(sf_losers) > 0  else "TBD"
    third_b    = sf_losers[1]  if len(sf_losers) > 1  else "TBD"

    final_match      = play(finalist_a, finalist_b)
    third_match      = play(third_a, third_b)
    champion         = final_match["winner"]
    third_place      = third_match["winner"]

    print(f"\n── Finale ───────────────────────────────────")
    print(f"  {finalist_a:<22} vs {finalist_b:<22} → 🏆 {champion}")
    print(f"\n── Troostfinale ─────────────────────────────")
    print(f"  {third_a:<22} vs {third_b:<22} → {third_place}")

    return {
        "r32_matchups": r32_matchups,
        "r16_matchups": r16_matchups,
        "qf_matchups":  qf_matchups,
        "sf_matchups":  sf_matchups,
        "final_match":  final_match,
        "third_match":  third_match,
        # legacy lists (voor backward compat)
        "r32":          r32_winners,
        "r16":          r16_winners,
        "qf":           qf_winners,
        "sf":           sf_winners,
        "finalist_a":   finalist_a,
        "finalist_b":   finalist_b,
        "champion":     champion,
        "third":        third_place,
    }


if __name__ == "__main__":
    from engine.predictor import load_resources
    resources = load_resources()
    result = predict_bracket(resources)
    print(f"\n🏆 Voorspelde wereldkampioen: {result['champion']}")
    print(f"🥉 Derde plaats: {result['third']}")
