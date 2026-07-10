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

## ☁️ Deploy naar Streamlit Cloud

Streamlit Community Cloud draait de app op een **ephemeral filesystem** — elke
container-restart (redeploy, slaapstand na inactiviteit, crash) wist lokale
wijzigingen. `record_result()` commit + pusht daarom automatisch elke
wijziging aan `matches.csv`, `teams.csv`, `elo_ratings.csv` en (bij hertraining)
`model.pkl`/`scaler.pkl` terug naar deze repo (zie `engine/git_sync.py`) — de
repo zelf is zo de enige bron van waarheid, niet de container. Bij het
opstarten doet de app eerst een `git pull` (vóór `load_resources()`) zodat een
update vanaf een ander device (bijv. mobiel) altijd wordt meegenomen.

### Stappen

1. **Push deze repo naar GitHub** (als dat nog niet zo is) — `origin` moet
   naar jouw eigen repo wijzen, niet een fork-upstream.

2. **Genereer een fine-grained GitHub Personal Access Token**:
   - GitHub → Settings → Developer settings → Personal access tokens →
     **Fine-grained tokens** → *Generate new token*
   - **Repository access**: alleen deze repo (`suluco/wk2026-predictor`) —
     niet "All repositories"
   - **Permissions** → Repository permissions → **Contents**: `Read and write`
     (dit is de enige scope die nodig is; laat alle andere permissions op
     "No access")
   - Zet een vervaldatum (bijv. 90 dagen — verleng na de vakantie desnoods)
   - Kopieer het token direct na aanmaken — het is daarna niet meer zichtbaar

3. **Koppel de repo op [share.streamlit.io](https://share.streamlit.io)**:
   - *New app* → kies deze GitHub-repo, branch `main`, main file `app.py`

4. **Zet het secret** — in de app-instellingen op share.streamlit.io:
   *Settings → Secrets*, en voeg toe:
   ```toml
   GITHUB_TOKEN = "github_pat_xxxxxxxxxxxxxxxxxxxxxxxx"
   ```
   Zonder dit secret start de app nog gewoon op (leest de laatste staat die
   al in de repo staat), maar elke nieuwe uitslag-invoer geeft een expliciete
   foutmelding in de UI in plaats van stilzwijgend verloren te gaan.

5. **Deploy**. Test daarna vanaf mobiel: voer een uitslag in via
   "UITSLAG INVOEREN" — de app moet een groene bevestiging tonen. Check op
   GitHub dat er een nieuwe commit staat met message
   `Uitslag bijgewerkt: {match_id} - {timestamp}`.

### Let op

- Elke uitslag-invoer met hertraining commit ook `model.pkl` (~5MB) — de
  git-geschiedenis groeit dus met elke invoer. Voor dit toernooi (±20-30
  resultaat-invoeren) is dat geen probleem; voor langdurig/veelvuldig gebruik
  is een externe modelopslag (bijv. alleen hertrainen na de groepsfase in
  plaats van na elke wedstrijd) beter houdbaar.
- Bij gelijktijdige invoer vanaf twee devices kan de push conflicteren (git
  rebase-fout) — de UI toont dit dan als expliciete foutmelding. Ververs de
  pagina (nieuwe `git pull`) en probeer opnieuw.

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