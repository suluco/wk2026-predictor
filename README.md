# ⚽ WK 2026 Voorspeller

Persoonlijk voorspelalgoritme voor het FIFA Wereldkampioenschap 2026 (Canada/VS/Mexico).  
Gebouwd voor gebruik in een poule met vrienden.

---

## 🧠 Hoe werkt het model?

De voorspellingen zijn gebaseerd op drie lagen die gecombineerd worden:

### Laag 1 — Poisson Monte Carlo simulatie
Elke wedstrijd wordt 50.000 keer gesimuleerd op basis van verwachte doelpunten
per team. Aanvalskracht × verdedigingsfactor van de tegenstander bepaalt de
Poisson-lambda. Geeft exacte score-distributies en meest waarschijnlijke uitslag.

### Laag 2 — XGBoost ML-model
Getraind op 192 WK-wedstrijden (2014, 2018, 2022). Features:
- Elo-ratingverschil
- Aanvals- en verdedigingsverschil
- Vormdifferentie
- WK-ervaringsverschil
- Verwachte doelpuntenverschil (lambda)
- Confederatie (zelfde of niet)

CV-accuraatheid: ~54.7% (baseline altijd A wint: 40.6%)

### Laag 3 — H2H correctie
Head-to-head historiek uit WK 2014/2018/2022. Maximaal 15% invloed,
opbouwend per gespeeld duel (5% per duel). Voorkomt overfitting op
kleine H2H-samples.

### Blend
- 45% XGBoost ML
- 55% Poisson Monte Carlo
- H2H als zachte correctie bovenop de blend

---

## 📊 Bayesiaanse updates

Na elke gespeelde wedstrijd (invoeren via tab "Uitslag invoeren"):
1. `matches.csv` wordt bijgewerkt
2. Teamsterktes (attack/defense/form) worden herberekend
3. Elo-ratings worden bijgewerkt
4. XGBoost model wordt hertraind op nieuwe data

Het model wordt dus beter naarmate het toernooi vordert.

---

## 🗂️ Projectstructuur
```
wk2026-predictor/
├── app.py                  # Streamlit UI
├── data/
│   ├── teams.csv           # Teamsterktes + FIFA ranking
│   ├── matches.csv         # Officieel speelschema + uitslagen
│   ├── results.csv         # Backup uitslagen
│   ├── elo_ratings.csv     # Elo-ratings (gegenereerd)
│   ├── h2h.csv             # Head-to-head historiek (gegenereerd)
│   ├── training_data.csv   # ML trainingsdata (gegenereerd)
│   ├── wc2014.json         # Historische WK-data
│   ├── wc2018.json         # Historische WK-data
│   └── wc2022.json         # Historische WK-data
└── engine/
    ├── simulator.py        # Poisson Monte Carlo kern
    ├── ratings.py          # Bayesiaanse updates na uitslagen
    ├── predictor.py        # Volledige voorspelpipeline
    ├── elo.py              # Elo-ratingsysteem
    ├── h2h.py              # Head-to-head statistieken
    ├── features.py         # Feature engineering voor ML
    ├── ml_model.py         # XGBoost training + voorspelling
    └── bracket.py          # Knock-out bracket simulatie
```

---

## 🚀 Installatie
```bash
# Clone de repo
git clone https://github.com/JOUWUSERNAME/wk2026-predictor.git
cd wk2026-predictor

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Dependencies
pip install -r requirements.txt

# Mac: XGBoost vereist libomp
brew install libomp

# App starten
streamlit run app.py
```

---

## 📅 Gebruik tijdens het toernooi

1. Open de app: `streamlit run app.py`
2. **Tab 1** — Selecteer een wedstrijd en krijg voorspelling + kansen
3. **Tab 2** — Bekijk alle voorspellingen per groep
4. **Tab 3** — Voer gespeelde uitslagen in na elke wedstrijd
5. **Tab 4** — Simuleer het volledige toernooi bracket

---

## 🔢 Data bronnen

| Data | Bron |
|------|------|
| Historische WK-data | openfootball/worldcup.json (open source) |
| FIFA ranking | Handmatig samengesteld (maart 2026) |
| Teamsterktes | Geschat op basis van FIFA ranking + WK-historiek |
| Speelschema WK 2026 | Officieel FIFA schema |

---

## ⚠️ Beperkingen

- Voetbal is inherent onvoorspelbaar. Zelfs de beste modellen halen ~55-60% accuraatheid.
- Spelersdata (blessures, rode kaarten, opstelling) zit niet in het model.
- TBD-teams (play-off winnaars) worden behandeld als gemiddeld team.
- H2H data is beperkt tot WK 2014/2018/2022 (3 toernooien).

---

*Gebouwd met Python · Streamlit · XGBoost · Scipy · Pandas*