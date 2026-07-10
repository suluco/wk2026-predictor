import streamlit as st
import pandas as pd
from pathlib import Path
from engine.predictor import predict_match, predict_all_upcoming, load_resources
from engine.ratings import record_result
from engine.git_sync import git_pull, git_commit_and_push, GitSyncError, timestamp

st.set_page_config(
    page_title="WK 2026 Voorspeller",
    page_icon="⚽",
    layout="centered",  # mobielvriendelijk — "wide" duwt kolommen te smal op een telefoonscherm
    initial_sidebar_state="collapsed"
)

DATA_DIR = Path("data")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');
:root{--green:#00C853;--dark:#090909;--card:#111;--border:#1e1e1e;--text:#e8e8e8;--muted:#4a4a4a}
html,body,[class*="css"]{background:var(--dark)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif}
h1,h2,h3,h4{font-family:'Bebas Neue',sans-serif!important;letter-spacing:2px}
.stButton>button{background:var(--green)!important;color:#000!important;font-family:'Bebas Neue',sans-serif!important;font-size:1rem!important;letter-spacing:2px!important;border:none!important;border-radius:6px!important;width:100%}
.stSelectbox>div>div{background:var(--card)!important;border:1px solid var(--border)!important;color:var(--text)!important}
.result-box{background:#0a1a0a;border:1px solid var(--green);border-radius:12px;padding:1.5rem;text-align:center;margin:1rem 0}
.score-big{font-family:'Bebas Neue',sans-serif;font-size:4.5rem;color:var(--green);letter-spacing:6px;line-height:1}
.prob-bg{background:var(--border);border-radius:4px;height:8px;margin:4px 0 12px;overflow:hidden}
.prob-fill{height:100%;border-radius:4px;background:var(--green)}
.muted{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:1px}
.reason-item{background:var(--card);border-left:3px solid var(--green);padding:.4rem .8rem;margin:.3rem 0;border-radius:0 6px 6px 0;font-size:.9rem}
.model-tag{background:#0a1a0a;border:1px solid var(--border);border-radius:4px;padding:.2rem .5rem;font-size:.75rem;color:var(--muted);margin:.2rem}
div[data-testid="stTabs"] button{font-family:'Bebas Neue',sans-serif!important;letter-spacing:1px!important}
</style>
""", unsafe_allow_html=True)

FLAGS = {
    "Argentina":"🇦🇷","Australia":"🇦🇺","Austria":"🇦🇹","Algeria":"🇩🇿",
    "Belgium":"🇧🇪","Brazil":"🇧🇷","Canada":"🇨🇦","Cape Verde":"🇨🇻",
    "Colombia":"🇨🇴","Croatia":"🇭🇷","Curaçao":"🇨🇼","Ecuador":"🇪🇨",
    "Egypt":"🇪🇬","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","France":"🇫🇷","Germany":"🇩🇪",
    "Ghana":"🇬🇭","Haiti":"🇭🇹","Iran":"🇮🇷","Ivory Coast":"🇨🇮",
    "Japan":"🇯🇵","Jordan":"🇯🇴","Mexico":"🇲🇽","Morocco":"🇲🇦",
    "Netherlands":"🇳🇱","New Zealand":"🇳🇿","Norway":"🇳🇴","Panama":"🇵🇦",
    "Paraguay":"🇵🇾","Portugal":"🇵🇹","Qatar":"🇶🇦","Saudi Arabia":"🇸🇦",
    "Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Senegal":"🇸🇳","South Africa":"🇿🇦","South Korea":"🇰🇷",
    "Spain":"🇪🇸","Switzerland":"🇨🇭","Tunisia":"🇹🇳","Uruguay":"🇺🇾",
    "USA":"🇺🇸","Uzbekistan":"🇺🇿","TBD":"🏳️",
}

def flag(t): return FLAGS.get(t, "🏳️")
def pbar(pct, color="var(--green)"):
    return f'<div class="prob-bg"><div class="prob-fill" style="width:{pct}%;background:{color}"></div></div>'

@st.cache_resource
def _startup_git_pull():
    """
    Draait precies één keer per container-lifetime (cache_resource), vóór
    de eerste load_resources(). Haalt de laatste staat van matches.csv/
    teams.csv/elo_ratings.csv/model.pkl op — nodig omdat Streamlit Cloud een
    ephemeral filesystem heeft en iemand vanaf een ander device (bijv.
    mobiel) intussen een uitslag kan hebben ingevoerd en gepusht.
    """
    git_pull()
    return True

_startup_git_pull()

@st.cache_data(ttl=60)
def get_matches():
    return pd.read_csv(DATA_DIR / "matches.csv")

@st.cache_resource
def get_resources():
    return load_resources()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 style="font-size:2.8rem;margin-bottom:0">⚽ WK 2026 VOORSPELLER</h1>', unsafe_allow_html=True)
st.markdown('<p class="muted" style="margin-top:0">Poisson Monte Carlo 50% · XGBoost ML 30% · Elo 20% · H2H correctie · Bayesiaanse updates</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["WEDSTRIJD VOORSPELLEN", "ALLE WEDSTRIJDEN", "UITSLAG INVOEREN", "TOERNOOI BRACKET", "TOPSCORERS"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Wedstrijd voorspellen
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    matches_df = get_matches()
    resources  = get_resources()

    upcoming = matches_df[
        (matches_df["played"] == 0) &
        (matches_df["home"] != "TBD") &
        (matches_df["away"] != "TBD")
    ].copy()

    def match_label(row):
        stage = str(row.get("stage", "group"))
        if stage != "group":
            prefix = stage
        else:
            prefix = f"Groep {row['group']}"
        return f"{prefix} | {row['date']} | {flag(row['home'])} {row['home']} vs {flag(row['away'])} {row['away']}"

    match_labels = {
        row["match_id"]: match_label(row)
        for _, row in upcoming.iterrows()
    }

    col1, col2 = st.columns([3, 1])
    with col1:
        chosen_id = st.selectbox(
            "Selecteer wedstrijd",
            options=list(match_labels.keys()),
            format_func=lambda x: match_labels[x]
        )
    with col2:
        knockout = st.checkbox("Knock-out fase", value=False)
        n_sim    = st.select_slider("Simulaties", [10_000, 25_000, 50_000, 100_000], value=50_000)

    if st.button("⚡ VOORSPEL"):
        row  = upcoming[upcoming["match_id"] == chosen_id].iloc[0]
        home, away = row["home"], row["away"]

        with st.spinner("Simuleren..."):
            result = predict_match(
                home, away, knockout=knockout, n=n_sim, resources=resources,
                match_id=int(row["match_id"]), match_date=str(row["date"]),
            )

        ml = result["most_likely_score"]
        fh, fa = flag(home), flag(away)

        # ── Score box ─────────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-box">
            <div class="muted">meest waarschijnlijke uitslag</div>
            <div style="display:flex;align-items:center;justify-content:center;gap:2.5rem;margin:.8rem 0">
                <div>
                    <div style="font-size:3rem">{fh}</div>
                    <div style="font-family:'Bebas Neue';font-size:1.1rem;letter-spacing:1px">{home}</div>
                </div>
                <div class="score-big">{ml[0]} – {ml[1]}</div>
                <div>
                    <div style="font-size:3rem">{fa}</div>
                    <div style="font-family:'Bebas Neue';font-size:1.1rem;letter-spacing:1px">{away}</div>
                </div>
            </div>
            <div style="font-family:'Bebas Neue';font-size:1.5rem;color:var(--green)">
                Voorspelling: {result['winner']}
            </div>
            <div class="muted" style="margin-top:.4rem">
                Zekerheid {result['confidence']}% &nbsp;·&nbsp;
                Verwacht {result['exp_goals_home']} – {result['exp_goals_away']} goals
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Model breakdown ───────────────────────────────────────────
        odds_active = result.get("odds_active", False)
        breakdown_label = "📊 Model breakdown (ML · Poisson · Elo · Odds)" if odds_active else "📊 Model breakdown (ML · Poisson · Elo)"
        with st.expander(breakdown_label):
            mp = result["ml_probs"]
            pp = result["poisson_probs"]
            ep = result["elo_probs"]
            op = result.get("odds_probs")

            if odds_active and op:
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.markdown("**XGBoost ML** (20%)")
                    st.markdown(f"Win {home}: **{round(mp['win_a']*100,1)}%**")
                    st.markdown(f"Gelijkspel: **{round(mp['draw']*100,1)}%**")
                    st.markdown(f"Win {away}: **{round(mp['win_b']*100,1)}%**")
                with mc2:
                    st.markdown("**Poisson MC** (35%)")
                    st.markdown(f"Win {home}: **{round(pp['win_a']*100,1)}%**")
                    st.markdown(f"Gelijkspel: **{round(pp['draw']*100,1)}%**")
                    st.markdown(f"Win {away}: **{round(pp['win_b']*100,1)}%**")
                with mc3:
                    st.markdown("**Elo** (15%)")
                    st.markdown(f"Win {home}: **{round(ep['win_a']*100,1)}%**")
                    st.markdown(f"Gelijkspel: **{round(ep['draw']*100,1)}%**")
                    st.markdown(f"Win {away}: **{round(ep['win_b']*100,1)}%**")
                with mc4:
                    n_books = op.get("bookmakers", "?")
                    st.markdown(f"**Bookmaker-odds** (30%) 📈")
                    st.markdown(f"Win {home}: **{round(op['win_a']*100,1)}%**")
                    st.markdown(f"Gelijkspel: **{round(op['draw']*100,1)}%**")
                    st.markdown(f"Win {away}: **{round(op['win_b']*100,1)}%**")
                    st.caption(f"{n_books} bookmakers gemiddeld")
            else:
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.markdown("**XGBoost ML** (30%)")
                    st.markdown(f"Win {home}: **{round(mp['win_a']*100,1)}%**")
                    st.markdown(f"Gelijkspel: **{round(mp['draw']*100,1)}%**")
                    st.markdown(f"Win {away}: **{round(mp['win_b']*100,1)}%**")
                with mc2:
                    st.markdown("**Poisson MC** (50%)")
                    st.markdown(f"Win {home}: **{round(pp['win_a']*100,1)}%**")
                    st.markdown(f"Gelijkspel: **{round(pp['draw']*100,1)}%**")
                    st.markdown(f"Win {away}: **{round(pp['win_b']*100,1)}%**")
                with mc3:
                    st.markdown("**Elo** (20%)")
                    st.markdown(f"Win {home}: **{round(ep['win_a']*100,1)}%**")
                    st.markdown(f"Gelijkspel: **{round(ep['draw']*100,1)}%**")
                    st.markdown(f"Win {away}: **{round(ep['win_b']*100,1)}%**")

        # ── Kansen ───────────────────────────────────────────────────
        st.markdown("#### KANSEN")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="muted">{fh} {home} wint</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-family:Bebas Neue;font-size:2rem;color:var(--green)">{result["win_home"]}%</div>', unsafe_allow_html=True)
            st.markdown(pbar(result["win_home"]), unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="muted">Gelijkspel</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-family:Bebas Neue;font-size:2rem;color:#555">{result["draw"]}%</div>', unsafe_allow_html=True)
            st.markdown(pbar(result["draw"], "#333"), unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="muted">{fa} {away} wint</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-family:Bebas Neue;font-size:2rem;color:var(--green)">{result["win_away"]}%</div>', unsafe_allow_html=True)
            st.markdown(pbar(result["win_away"]), unsafe_allow_html=True)

        # ── Uitleg + top scores ───────────────────────────────────────
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### WAAROM DEZE VOORSPELLING")
            for reason in result["reasons"]:
                st.markdown(f'<div class="reason-item">• {reason}</div>', unsafe_allow_html=True)
        with col_r:
            st.markdown("#### TOP UITSLAGEN")
            for score, pct in result["top_scores"]:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:.8rem;margin:.3rem 0">
                    <div style="font-family:'Bebas Neue';font-size:1.1rem;width:3rem;text-align:right">{score}</div>
                    <div class="prob-bg" style="flex:1;margin:0">
                        <div class="prob-fill" style="width:{min(pct*6,100)}%"></div>
                    </div>
                    <div style="font-size:.85rem;color:var(--muted);width:3rem">{pct}%</div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Alle wedstrijden
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### ALLE AANKOMENDE WEDSTRIJDEN")
    st.markdown('<p class="muted">Klik op de knop om alle wedstrijden te simuleren via het volledige model.</p>', unsafe_allow_html=True)

    if st.button("🔄 Genereer alle voorspellingen"):
        with st.spinner("Alle wedstrijden simuleren..."):
            all_preds = predict_all_upcoming(n=25_000)

        if all_preds.empty:
            st.info("Geen wedstrijden meer te voorspellen.")
        else:
            all_preds["match"] = all_preds.apply(
                lambda r: f"{flag(r['home'])} {r['home']} vs {flag(r['away'])} {r['away']}", axis=1
            )
            all_preds["score"] = all_preds["pred_home"].astype(str) + "–" + all_preds["pred_away"].astype(str)

            STAGE_LABELS = {"R32": "ROUND OF 32", "R16": "ROUND OF 16", "QF": "KWARTFINALES", "SF": "HALVE FINALES", "F": "FINALE"}
            for group in sorted(all_preds["group"].unique()):
                label = STAGE_LABELS.get(group, f"GROEP {group}")
                st.markdown(f"**{label}**")
                grp = all_preds[all_preds["group"] == group][[
                    "date","match","score","win_home","draw","win_away","winner","confidence"
                ]].rename(columns={
                    "date":"Datum","match":"Wedstrijd","score":"Pred. score",
                    "win_home":"Thuis%","draw":"Gelijk%","win_away":"Uit%",
                    "winner":"Winnaar","confidence":"Zekerheid%"
                })
                st.dataframe(grp, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Uitslag invoeren
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### GESPEELDE UITSLAG INVOEREN")
    st.markdown('<p class="muted">Na elke wedstrijd voer je de uitslag in — Elo en teamsterktes worden automatisch bijgewerkt.</p>', unsafe_allow_html=True)

    matches_df3 = get_matches()

    # Succesmelding overleeft de st.rerun() na het opslaan: st.success() vóór
    # st.rerun() wordt door de onmiddellijke herstart weggevaagd voordat de
    # gebruiker hem ziet, dus de boodschap wordt in session_state gezet en na
    # de rerun hier getoond + meteen weer verwijderd (toont maar één keer).
    pending_success = st.session_state.pop("result_save_success", None)
    if pending_success:
        st.success(pending_success)

    # ── Sectie A: strafschoppen invoeren voor gespeeld gelijkspel in KO ───────
    if "penalty_winner" in matches_df3.columns:
        ko_tied = matches_df3[
            (matches_df3["played"] == 1) &
            (matches_df3["stage"] != "group") &
            (matches_df3["home_score"] == matches_df3["away_score"]) &
            (matches_df3["penalty_winner"].fillna("").astype(str).str.strip() == "")
        ]
    else:
        ko_tied = pd.DataFrame()

    if not ko_tied.empty:
        st.warning(f"⚠️ {len(ko_tied)} knockout-wedstrijd(en) eindigden gelijk — voeg de strafschoppenwinnaar toe:")
        for _, trow in ko_tied.iterrows():
            with st.form(key=f"pk_form_{trow['match_id']}"):
                st.markdown(f"**{flag(trow['home'])} {trow['home']} {int(trow['home_score'])}–{int(trow['away_score'])} {flag(trow['away'])} {trow['away']}** (na verlengingen)")
                pk_winner = st.radio(
                    "Winnaar na strafschoppen",
                    [trow["home"], trow["away"]],
                    horizontal=True,
                    key=f"pk_{trow['match_id']}"
                )
                if st.form_submit_button("✅ OPSLAAN"):
                    try:
                        record_result(int(trow["match_id"]),
                                      int(trow["home_score"]),
                                      int(trow["away_score"]),
                                      penalty_winner=str(pk_winner))
                        from engine.auto_updater import propagate_knockout_fixtures
                        added = propagate_knockout_fixtures()
                        if added:
                            git_commit_and_push(
                                ["data/matches.csv"],
                                message=f"Knockout fixture toegevoegd - {timestamp()}",
                            )
                    except GitSyncError as e:
                        st.error(f"⚠️ Uitslag is lokaal opgeslagen maar NIET gepersisteerd naar git: {e}")
                    else:
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        msg = f"✅ {pk_winner} als winnaar na strafschoppen opgeslagen en gesynchroniseerd naar GitHub."
                        if added:
                            msg += f" {added} nieuwe wedstrijd{'en' if added != 1 else ''} toegevoegd aan de bracket."
                        st.session_state["result_save_success"] = msg
                        st.rerun()

        st.divider()

    # ── Sectie B: nieuwe uitslag invoeren ─────────────────────────────────────
    unplayed = matches_df3[
        (matches_df3["played"] == 0) &
        (matches_df3["home"] != "TBD") &
        (matches_df3["away"] != "TBD")
    ]

    if unplayed.empty:
        st.success("Alle wedstrijden zijn ingevoerd!")
    else:
        result_labels = {
            row["match_id"]: f"{row['date']} | {flag(row['home'])} {row['home']} vs {flag(row['away'])} {row['away']}"
            for _, row in unplayed.iterrows()
        }

        sel_id  = st.selectbox("Wedstrijd", list(result_labels.keys()), format_func=lambda x: result_labels[x])
        sel_row = unplayed[unplayed["match_id"] == sel_id].iloc[0]
        is_knockout = str(sel_row.get("stage", "group")) != "group"

        rc1, rc2, rc3 = st.columns([2, 1, 2])
        with rc1:
            st.markdown(f"**{flag(sel_row['home'])} {sel_row['home']}**")
            home_score = st.number_input("Goals thuis", min_value=0, max_value=20, value=0, key="hs")
        with rc2:
            st.markdown("<br><div style='text-align:center;font-family:Bebas Neue;font-size:1.5rem;color:#333'>–</div>", unsafe_allow_html=True)
        with rc3:
            st.markdown(f"**{flag(sel_row['away'])} {sel_row['away']}**")
            away_score = st.number_input("Goals uit", min_value=0, max_value=20, value=0, key="as")

        # Strafschoppen selector alleen zichtbaar als KO + gelijkspel
        pen_winner = ""
        if is_knockout and home_score == away_score:
            st.markdown('<p class="muted">Gelijkspel in knockout-fase — wie wint de strafschoppen?</p>', unsafe_allow_html=True)
            pen_winner = st.radio(
                "Winnaar na strafschoppen",
                [sel_row["home"], sel_row["away"]],
                horizontal=True,
                key="pen_winner_new"
            )

        if st.button("✅ UITSLAG OPSLAAN & MODEL UPDATEN"):
            try:
                record_result(int(sel_id), int(home_score), int(away_score), penalty_winner=pen_winner)
                added = 0
                if is_knockout:
                    from engine.auto_updater import propagate_knockout_fixtures
                    added = propagate_knockout_fixtures()
                    if added:
                        git_commit_and_push(
                            ["data/matches.csv"],
                            message=f"Knockout fixture toegevoegd - {timestamp()}",
                        )
            except GitSyncError as e:
                st.error(f"⚠️ Uitslag is lokaal opgeslagen maar NIET gepersisteerd naar git: {e}")
            else:
                st.cache_data.clear()
                st.cache_resource.clear()
                msg = f"✅ {sel_row['home']} {home_score}–{away_score} {sel_row['away']} opgeslagen en gesynchroniseerd naar GitHub."
                if added:
                    msg += f" {added} nieuwe wedstrijd{'en' if added != 1 else ''} toegevoegd aan de bracket."
                st.session_state["result_save_success"] = msg
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Toernooi bracket
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### TOERNOOI VOORSPELLING")
    st.markdown('<p class="muted">Simuleert het volledige toernooi op basis van huidige teamsterktes en Elo-ratings.</p>', unsafe_allow_html=True)

    st.info("⚠️ Bracket wordt berekend op basis van Elo-ranking voor ongespeelde groepswedstrijden. Hoe meer uitslagen je invoert, hoe accurater de bracket.")

    if st.button("🏆 SIMULEER VOLLEDIG TOERNOOI"):
        from engine.bracket import predict_bracket

        with st.spinner("Toernooi simuleren... (dit duurt ~30 seconden)"):
            resources = get_resources()
            bracket   = predict_bracket(resources)

        # Visualisatie
        def render_matchup(m: dict):
            a, b, w = m["team_a"], m["team_b"], m["winner"]
            sc = m["score"]
            wa, wb = m.get("win_a", 0), m.get("win_b", 0)
            fa, fb, fw = flag(a), flag(b), flag(w)
            winner_side = "left" if w == a else "right"
            aw = "font-weight:700;color:var(--green)" if winner_side == "left" else "color:var(--muted)"
            bw = "font-weight:700;color:var(--green)" if winner_side == "right" else "color:var(--muted)"
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);border-radius:8px;
                        padding:.6rem 1rem;margin:.25rem 0;display:flex;align-items:center;gap:.6rem">
                <div style="flex:1;text-align:right;{aw}">{fa} {a}<br>
                    <span style="font-size:.72rem;color:var(--muted)">{wa}%</span></div>
                <div style="font-family:'Bebas Neue';font-size:1.4rem;color:var(--green);
                            min-width:4rem;text-align:center">{sc[0]}–{sc[1]}</div>
                <div style="flex:1;text-align:left;{bw}">{fb} {b}<br>
                    <span style="font-size:.72rem;color:var(--muted)">{wb}%</span></div>
                <div style="font-size:.8rem;border-left:1px solid var(--border);
                            padding-left:.6rem;min-width:7rem">→ {fw} {w}</div>
            </div>
            """, unsafe_allow_html=True)

        round_defs = [
            ("Round of 32",   "r32_matchups",  2),
            ("Round of 16",   "r16_matchups",  2),
            ("Kwartfinales",  "qf_matchups",   1),
            ("Halve finales", "sf_matchups",   1),
        ]

        for round_name, key, ncols in round_defs:
            st.markdown(f"**{round_name}**")
            matchups = bracket.get(key, [])
            if ncols == 2 and len(matchups) >= 2:
                half = len(matchups) // 2
                col_left, col_right = st.columns(2)
                for i, m in enumerate(matchups):
                    with (col_left if i < half else col_right):
                        render_matchup(m)
            else:
                for m in matchups:
                    render_matchup(m)

        # Finalisten
        st.markdown("---")
        fm  = bracket.get("final_match", {})
        tm  = bracket.get("third_match", {})
        fa  = bracket["finalist_a"]
        fb  = bracket["finalist_b"]
        champ = bracket["champion"]
        third = bracket["third"]
        fsc = fm.get("score", (0, 0))
        tsc = tm.get("score", (0, 0))
        t3a = tm.get("team_a", "TBD")
        t3b = tm.get("team_b", "TBD")

        st.markdown(f"""
        <div class="result-box" style="margin-top:1rem">
            <div class="muted">voorspelde finale</div>
            <div style="display:flex;align-items:center;justify-content:center;gap:2rem;margin:.8rem 0">
                <div>
                    <div style="font-size:2.5rem">{flag(fa)}</div>
                    <div style="font-family:'Bebas Neue';font-size:1rem">{fa}</div>
                    <div class="muted">{fm.get('win_a',0)}%</div>
                </div>
                <div style="font-family:'Bebas Neue';font-size:2rem;color:var(--green)">{fsc[0]}–{fsc[1]}</div>
                <div>
                    <div style="font-size:2.5rem">{flag(fb)}</div>
                    <div style="font-family:'Bebas Neue';font-size:1rem">{fb}</div>
                    <div class="muted">{fm.get('win_b',0)}%</div>
                </div>
            </div>
            <div style="font-family:'Bebas Neue';font-size:2rem;color:var(--green)">
                🏆 {flag(champ)} {champ}
            </div>
            <div class="muted" style="margin-top:.5rem">
                🥉 Derde plaats: {flag(t3a)} {t3a} {tsc[0]}–{tsc[1]} {flag(t3b)} {t3b} → {flag(third)} {third}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Topscorers
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### TOPSCORERS VOORSPELLING")
    st.markdown(
        '<p class="muted">Scorekans per wedstrijd via Poisson (λ = doelpunten/90 uit groepsfase), '
        'gecorrigeerd voor tegenstander via Elo-ratings. '
        'Verwacht aantal wedstrijden berekend op basis van Elo-winkans per ronde.</p>',
        unsafe_allow_html=True
    )

    if st.button("⚽ VOORSPEL TOPSCORERS"):
        from engine.top_scorers import predict_top_scorers

        with st.spinner("Topscorers berekenen..."):
            ts_resources = get_resources()
            scorers = predict_top_scorers(ts_resources, top_n=10)

        if not scorers:
            st.info("Geen spelersdata beschikbaar.")
        else:
            # Gouden schoen favoriet apart uitgelicht
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a1200,#111);
                        border:1px solid #c8a000;border-radius:12px;
                        padding:1rem 1.2rem;margin-bottom:1rem">
                <div class="muted" style="color:#c8a000">🥇 GOUDEN SCHOEN FAVORIET</div>
                <div style="font-family:'Bebas Neue';font-size:1.6rem;letter-spacing:1px;margin:.3rem 0">
                    {flag(scorers[0]['team'])} {scorers[0]['name']}
                    <span style="color:#c8a000;font-size:1.1rem">
                        &nbsp;·&nbsp; {scorers[0]['goals_so_far']} goals
                    </span>
                </div>
                <div style="display:flex;gap:2rem;flex-wrap:wrap;font-size:.85rem">
                    <span><span class="muted">team</span>&nbsp; {scorers[0]['team']}</span>
                    <span><span class="muted">λ/90</span>&nbsp; {scorers[0]['goals_per_90']}</span>
                    <span><span class="muted">scorekans volgende wedstrijd</span>&nbsp;
                        <span style="color:var(--green);font-weight:700">
                            {scorers[0]['score_probability_per_match']}%
                        </span>
                        &nbsp;vs {flag(scorers[0]['next_opponent'])} {scorers[0]['next_opponent']}
                    </span>
                    <span><span class="muted">verwachte wedstrijden</span>&nbsp;
                        {scorers[0]['matches_remaining']}
                    </span>
                    <span><span class="muted">geprojecteerde goals</span>&nbsp;
                        <span style="color:var(--green)">{scorers[0]['projected_goals']}</span>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Ranking tabel (alle 10)
            st.markdown("**TOP 10 SCORERS — REST VAN HET TOERNOOI**")
            for rank, p in enumerate(scorers, start=1):
                medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"**{rank}.**")
                bar_w = min(p["score_probability_per_match"] * 1.5, 100)
                st.markdown(f"""
                <div style="background:var(--card);border:1px solid var(--border);border-radius:8px;
                            padding:.55rem 1rem;margin:.2rem 0;
                            display:flex;align-items:center;gap:.8rem">
                    <div style="min-width:2rem;text-align:center;font-family:'Bebas Neue';
                                font-size:1rem;color:var(--muted)">{medal}</div>
                    <div style="min-width:12rem">
                        <div style="font-weight:600">{flag(p['team'])} {p['name']}</div>
                        <div style="font-size:.75rem;color:var(--muted)">{p['team']}</div>
                    </div>
                    <div style="min-width:4rem;text-align:center">
                        <div style="font-family:'Bebas Neue';font-size:1.3rem;color:var(--green)">{p['goals_so_far']}</div>
                        <div style="font-size:.7rem;color:var(--muted)">goals</div>
                    </div>
                    <div style="flex:1">
                        <div style="font-size:.72rem;color:var(--muted);margin-bottom:2px">
                            scorekans vs {flag(p['next_opponent'])} {p['next_opponent']}
                        </div>
                        <div style="display:flex;align-items:center;gap:.5rem">
                            <div class="prob-bg" style="flex:1;margin:0">
                                <div class="prob-fill" style="width:{bar_w}%"></div>
                            </div>
                            <div style="font-size:.85rem;font-weight:600;
                                        color:var(--green);min-width:3rem">{p['score_probability_per_match']}%</div>
                        </div>
                    </div>
                    <div style="min-width:5rem;text-align:right;font-size:.82rem;color:var(--muted)">
                        ~{p['matches_remaining']} wedstr.<br>
                        <span style="color:var(--text)">{p['projected_goals']} proj.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(
                '<p class="muted" style="margin-top:.8rem;font-size:.7rem">'
                'Spelersdata groepsfase: statisch bijgehouden. '
                'Scorekans gecorrigeerd voor tegenstander via Elo. '
                'Geprojecteerde goals = λ_adj × verwachte wedstrijden. '
                'Werkelijke tournament data kan afwijken.'
                '</p>',
                unsafe_allow_html=True
            )