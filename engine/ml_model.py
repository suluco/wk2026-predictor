import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from engine.features import FEATURE_COLS

DATA_DIR  = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH  = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# ── Trainen ───────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame = None) -> tuple:
    """
    Train XGBoost classifier op historische WK-data.
    Labels: 0 = B wint, 1 = gelijkspel, 2 = A wint
    """
    if df is None:
        df = pd.read_csv(DATA_DIR / "training_data.csv")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(int)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # XGBoost met kalibratie (zorgt voor betrouwbare kansen)
    base_model = XGBClassifier(
        n_estimators=300,
        max_depth=3,           # Ondiep: voorkomt overfitting op kleine dataset
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

    # Kalibreer kansen via cross-validation (isotone regressie)
    model = CalibratedClassifierCV(base_model, cv=5, method="isotonic")
    model.fit(X_scaled, y)

    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")

    print(f"✓ Model getraind op {len(df)} wedstrijden")
    print(f"  CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  (Baseline = altijd A wint: {(y==2).mean():.3f})")

    # Opslaan
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ Model opgeslagen → {MODEL_PATH}")

    return model, scaler

def load_model() -> tuple:
    """Laad model en scaler. Train opnieuw als niet gevonden."""
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler

    print("Model niet gevonden, opnieuw trainen...")
    return train()

# ── Voorspellen ───────────────────────────────────────────────────────────────

def predict_proba(features_array: np.ndarray, model=None, scaler=None) -> dict:
    """
    Voorspel winkansen voor één wedstrijd via ML-model.
    Returns dict: win_a, draw, win_b (fracties, som = 1.0)
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    X = features_array.reshape(1, -1).astype(np.float32)
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0]

    # Classes: 0=B wint, 1=gelijkspel, 2=A wint
    classes = list(model.classes_)
    prob_dict = dict(zip(classes, proba))

    return {
        "win_a": round(float(prob_dict.get(2, 0.33)), 4),
        "draw":  round(float(prob_dict.get(1, 0.25)), 4),
        "win_b": round(float(prob_dict.get(0, 0.33)), 4),
    }

def blend_predictions(
    ml_probs: dict,
    poisson_probs: dict,
    h2h_stats: dict,
    elo_probs: dict = None,
    ml_weight: float = 0.30,
    poisson_weight: float = 0.50,
    elo_weight: float = 0.20,
) -> dict:
    """
    Blend ML + Poisson + Elo.
    Standaard gewichten: ML 30%, Poisson 50%, Elo 20%.
    H2H als zachte correctie bovenop de blend.

    Elo als derde component corrigeert historische ML-bias:
    het ML-model traint op kleine WK-dataset (192 matches) en kan
    teams over/onderschatten op basis van toernooi-succes in 2014-2022.
    """
    if elo_probs is None:
        # Geen Elo: herverdeel gewichten over ML + Poisson
        total = ml_weight + poisson_weight
        ml_w  = ml_weight / total
        poi_w = poisson_weight / total
        blended = {
            "win_a": ml_w * ml_probs["win_a"] + poi_w * poisson_probs["win_a"],
            "draw":  ml_w * ml_probs["draw"]  + poi_w * poisson_probs["draw"],
            "win_b": ml_w * ml_probs["win_b"] + poi_w * poisson_probs["win_b"],
        }
    else:
        blended = {
            "win_a": ml_weight * ml_probs["win_a"] + poisson_weight * poisson_probs["win_a"] + elo_weight * elo_probs["win_a"],
            "draw":  ml_weight * ml_probs["draw"]  + poisson_weight * poisson_probs["draw"]  + elo_weight * elo_probs["draw"],
            "win_b": ml_weight * ml_probs["win_b"] + poisson_weight * poisson_probs["win_b"] + elo_weight * elo_probs["win_b"],
        }

    # H2H correctie — alleen als er minstens 1 duel is
    games = h2h_stats.get("games", 0)
    if games > 0:
        h2h_weight = min(games * 0.05, 0.15)  # Max 15% invloed, opbouwend per duel
        h2h_win_a  = h2h_stats["win_ratio_a"]
        h2h_win_b  = 1 - h2h_win_a
        h2h_draw   = h2h_stats["draws"] / games

        # Normaliseer H2H kansen
        h2h_total  = h2h_win_a + h2h_draw + h2h_win_b
        h2h_win_a /= h2h_total
        h2h_draw  /= h2h_total
        h2h_win_b /= h2h_total

        rest = 1 - h2h_weight
        blended = {
            "win_a": rest * blended["win_a"] + h2h_weight * h2h_win_a,
            "draw":  rest * blended["draw"]  + h2h_weight * h2h_draw,
            "win_b": rest * blended["win_b"] + h2h_weight * h2h_win_b,
        }

    # Normaliseer zodat alles optelt tot 1.0
    total = sum(blended.values())
    return {k: round(v / total, 4) for k, v in blended.items()}

# ── Feature importance ────────────────────────────────────────────────────────

def print_feature_importance(model=None, scaler=None):
    if model is None:
        model, scaler = load_model()

    # Haal de onderliggende XGBoost op uit de CalibratedClassifierCV
    try:
        base = model.calibrated_classifiers_[0].estimator
        importances = base.feature_importances_
        pairs = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
        print("\nFeature importance (XGBoost):")
        for feat, imp in pairs:
            bar = "█" * int(imp * 100)
            print(f"  {feat:<22} {bar} {imp:.3f}")
    except Exception as e:
        print(f"Feature importance niet beschikbaar: {e}")

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from engine.features import build_features, features_to_array, build_training_data
    from engine.elo import load_elo
    from engine.h2h import load_h2h
    from engine.simulator import load_teams
    from engine.climate import load_stadiums
    import pandas as pd

    elo_dict    = load_elo()
    h2h_df      = load_h2h()
    teams_df    = load_teams()
    stadiums_df = load_stadiums()
    matches_df  = pd.read_csv("data/matches.csv")

    # Train
    df = build_training_data()
    model, scaler = train(df)
    print_feature_importance(model, scaler)

    # Test voorspellingen
    test_matches = [
        ("Netherlands", "Argentina", 10, "2026-06-14"),
        ("France",      "Brazil",    17, "2026-06-16"),
        ("Germany",     "Spain",     9,  "2026-06-14"),
        ("England",     "Morocco",   22, "2026-06-17"),
    ]

    print("\n── ML Voorspellingen ──────────────────────────────────")
    for a, b, mid, date in test_matches:
        feats = build_features(
            a, b, elo_dict, h2h_df, teams_df,
            match_id=mid, match_date=date,
            matches_df=matches_df, stadiums_df=stadiums_df,
        )
        arr   = features_to_array(feats)
        probs = predict_proba(arr, model, scaler)
        print(f"\n  {a} vs {b}")
        print(f"    Win {a:<15}: {round(probs['win_a']*100,1)}%")
        print(f"    Gelijkspel     : {round(probs['draw']*100,1)}%")
        print(f"    Win {b:<15}: {round(probs['win_b']*100,1)}%")