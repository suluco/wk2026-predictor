import streamlit as st
import pandas as pd
from pathlib import Path
from engine.predictor import predict_match, predict_all_upcoming, load_resources
from engine.ratings import record_result

st.set_page_config(
    page_title="WK 2026 Voorspeller",
    page_icon="⚽",
    layout="wide",
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

@st.cache_data(ttl=60)
def get_matches():
    return pd.read_csv(DATA_DIR / "matches.csv")

@st.cache_resource
def get_resources():
    return load_resources()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 style="font-size:2.8rem;margin-bottom:0">⚽ WK 2026 VOORSPELLER</h1>', unsafe_allow_html=True)
st.markdown('<p class="muted" style="margin-top:0">Poisson Monte Carlo · XGBoost ML · Elo · H2H · Bayesiaanse updates</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["WEDSTRIJD VOORSPELLEN", "ALLE WEDSTRIJDEN", "UITSLAG INVOEREN"])

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

    match_labels = {
        row["match_id"]: f"Groep {row['group']} | {row['date']} | {flag(row['home'])} {row['home']} vs {flag(row['away'])} {row['away']}"
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
            result = predict_match(home, away, knockout=knockout, n=n_sim, resources=resources)

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
        with st.expander("📊 Model breakdown (ML vs Poisson)"):
            mc1, mc2 = st.columns(2)
            mp = result["ml_probs"]
            pp = result["poisson_probs"]
            with mc1:
                st.markdown("**XGBoost ML**")
                st.markdown(f"Win {home}: **{round(mp['win_a']*100,1)}%**")
                st.markdown(f"Gelijkspel: **{round(mp['draw']*100,1)}%**")
                st.markdown(f"Win {away}: **{round(mp['win_b']*100,1)}%**")
            with mc2:
                st.markdown("**Poisson Monte Carlo**")
                st.markdown(f"Win {home}: **{round(pp['win_a']*100,1)}%**")
                st.markdown(f"Gelijkspel: **{round(pp['draw']*100,1)}%**")
                st.markdown(f"Win {away}: **{round(pp['win_b']*100,1)}%**")

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

            for group in sorted(all_preds["group"].unique()):
                st.markdown(f"**GROEP {group}**")
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

        rc1, rc2, rc3 = st.columns([2, 1, 2])
        with rc1:
            st.markdown(f"**{flag(sel_row['home'])} {sel_row['home']}**")
            home_score = st.number_input("Goals thuis", min_value=0, max_value=20, value=0, key="hs")
        with rc2:
            st.markdown("<br><div style='text-align:center;font-family:Bebas Neue;font-size:1.5rem;color:#333'>–</div>", unsafe_allow_html=True)
        with rc3:
            st.markdown(f"**{flag(sel_row['away'])} {sel_row['away']}**")
            away_score = st.number_input("Goals uit", min_value=0, max_value=20, value=0, key="as")

        if st.button("✅ UITSLAG OPSLAAN & MODEL UPDATEN"):
            record_result(int(sel_id), int(home_score), int(away_score))
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success(f"✓ {sel_row['home']} {home_score}–{away_score} {sel_row['away']} opgeslagen. Elo en teamsterktes bijgewerkt.")
            st.rerun()