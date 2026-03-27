import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"

# ── H2H opbouwen vanuit historische WK-data ───────────────────────────────────

def build_h2h() -> dict:
    """
    Bouw head-to-head statistieken op vanuit WK 2014, 2018, 2022.
    Returns dict: (team_a, team_b) -> {"wins_a": int, "draws": int, "wins_b": int, "goals_a": int, "goals_b": int}
    """
    h2h = defaultdict(lambda: {"wins_a": 0, "draws": 0, "wins_b": 0, "goals_a": 0, "goals_b": 0})

    for year in [2014, 2018, 2022]:
        path = DATA_DIR / f"wc{year}.json"
        if not path.exists():
            continue

        data = json.loads(path.read_text())

        for match in data.get("matches", []):
            team1 = match.get("team1", "")
            team2 = match.get("team2", "")
            score = match.get("score", {})

            if not score:
                continue

            ft = score.get("ft", None)
            if not ft or len(ft) < 2:
                continue

            g1, g2 = int(ft[0]), int(ft[1])

            # Altijd opslaan als gesorteerd paar zodat (A,B) == (B,A)
            key = tuple(sorted([team1, team2]))
            is_flipped = key[0] != team1

            if not is_flipped:
                h2h[key]["goals_a"] += g1
                h2h[key]["goals_b"] += g2
                if g1 > g2:
                    h2h[key]["wins_a"] += 1
                elif g1 == g2:
                    h2h[key]["draws"] += 1
                else:
                    h2h[key]["wins_b"] += 1
            else:
                h2h[key]["goals_a"] += g2
                h2h[key]["goals_b"] += g1
                if g2 > g1:
                    h2h[key]["wins_a"] += 1
                elif g1 == g2:
                    h2h[key]["draws"] += 1
                else:
                    h2h[key]["wins_b"] += 1

    return dict(h2h)

def save_h2h(h2h: dict):
    """Sla H2H op als CSV."""
    rows = []
    for (team_a, team_b), stats in h2h.items():
        rows.append({
            "team_a":  team_a,
            "team_b":  team_b,
            "wins_a":  stats["wins_a"],
            "draws":   stats["draws"],
            "wins_b":  stats["wins_b"],
            "goals_a": stats["goals_a"],
            "goals_b": stats["goals_b"],
        })
    pd.DataFrame(rows).to_csv(DATA_DIR / "h2h.csv", index=False)
    print(f"✓ h2h.csv opgeslagen ({len(rows)} matchups)")

def load_h2h() -> pd.DataFrame:
    path = DATA_DIR / "h2h.csv"
    if path.exists():
        return pd.read_csv(path)
    h2h = build_h2h()
    save_h2h(h2h)
    return load_h2h()

def get_h2h_stats(team_a: str, team_b: str, h2h_df: pd.DataFrame) -> dict:
    """
    Haal H2H stats op voor twee teams.
    Geeft altijd stats terug vanuit perspectief van team_a.
    """
    key = tuple(sorted([team_a, team_b]))
    flipped = key[0] != team_a

    mask = (h2h_df["team_a"] == key[0]) & (h2h_df["team_b"] == key[1])
    row  = h2h_df[mask]

    if row.empty:
        return {
            "games":       0,
            "wins_a":      0,
            "draws":       0,
            "wins_b":      0,
            "goals_a":     0,
            "goals_b":     0,
            "win_ratio_a": 0.5,  # Geen data: neutrale prior
        }

    r = row.iloc[0]
    games = int(r["wins_a"] + r["draws"] + r["wins_b"])

    if not flipped:
        wins_a  = int(r["wins_a"])
        wins_b  = int(r["wins_b"])
        goals_a = int(r["goals_a"])
        goals_b = int(r["goals_b"])
    else:
        wins_a  = int(r["wins_b"])
        wins_b  = int(r["wins_a"])
        goals_a = int(r["goals_b"])
        goals_b = int(r["goals_a"])

    draws = int(r["draws"])

    # Win ratio met Laplace smoothing (voorkomt 0% of 100%)
    win_ratio_a = (wins_a + 1) / (games + 2)

    return {
        "games":       games,
        "wins_a":      wins_a,
        "draws":       draws,
        "wins_b":      wins_b,
        "goals_a":     goals_a,
        "goals_b":     goals_b,
        "win_ratio_a": round(win_ratio_a, 4),
    }

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("H2H opbouwen vanuit WK 2014/2018/2022...")
    h2h = build_h2h()
    save_h2h(h2h)

    h2h_df = load_h2h()
    print(f"\nTotaal unieke matchups: {len(h2h_df)}")

    # Bekende confrontaties testen
    tests = [
        ("Brazil",   "Germany"),
        ("Argentina","Netherlands"),
        ("France",   "Croatia"),
        ("Spain",    "Netherlands"),
    ]

    print("\nBekende H2H's:")
    for a, b in tests:
        stats = get_h2h_stats(a, b, h2h_df)
        if stats["games"] > 0:
            print(f"\n  {a} vs {b} ({stats['games']} WK-duels):")
            print(f"    Winst {a:<15}: {stats['wins_a']}")
            print(f"    Gelijkspel     : {stats['draws']}")
            print(f"    Winst {b:<15}: {stats['wins_b']}")
            print(f"    Goals          : {stats['goals_a']}–{stats['goals_b']}")
            print(f"    Win ratio {a[:3]}  : {round(stats['win_ratio_a']*100,1)}%")
        else:
            print(f"\n  {a} vs {b}: geen WK-historiek")