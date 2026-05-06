import pandas as pd

UEFA_PATH_A = "Bosnia and Herzegovina"
UEFA_PATH_B = "Sweden"
UEFA_PATH_C = "Turkey"
UEFA_PATH_D = "Czechia"
IC_PATH_1   = "DR Congo"
IC_PATH_2   = "Iraq"

def update():
    df = pd.read_csv("data/matches.csv")
    original = df.copy()

    df.loc[(df["group"] == "A") & (df["home"] == "TBD"), "home"] = UEFA_PATH_D
    df.loc[(df["group"] == "A") & (df["away"] == "TBD"), "away"] = UEFA_PATH_D
    df.loc[(df["group"] == "B") & (df["home"] == "TBD"), "home"] = UEFA_PATH_A
    df.loc[(df["group"] == "B") & (df["away"] == "TBD"), "away"] = UEFA_PATH_A
    df.loc[(df["group"] == "D") & (df["home"] == "TBD"), "home"] = UEFA_PATH_C
    df.loc[(df["group"] == "D") & (df["away"] == "TBD"), "away"] = UEFA_PATH_C
    df.loc[(df["group"] == "F") & (df["home"] == "TBD"), "home"] = UEFA_PATH_B
    df.loc[(df["group"] == "F") & (df["away"] == "TBD"), "away"] = UEFA_PATH_B
    df.loc[(df["group"] == "I") & (df["home"] == "TBD"), "home"] = IC_PATH_2
    df.loc[(df["group"] == "I") & (df["away"] == "TBD"), "away"] = IC_PATH_2
    df.loc[(df["group"] == "K") & (df["home"] == "TBD"), "home"] = IC_PATH_1
    df.loc[(df["group"] == "K") & (df["away"] == "TBD"), "away"] = IC_PATH_1

    df.to_csv("data/matches.csv", index=False)

    changed = (df[["home","away"]] != original[["home","away"]]).any(axis=1)
    print(f"✓ {changed.sum()} wedstrijden bijgewerkt")
    for _, row in df[changed].iterrows():
        print(f"  Groep {row['group']} | {row['date']} | {row['home']} vs {row['away']}")

if __name__ == "__main__":
    update()
