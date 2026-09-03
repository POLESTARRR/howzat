"""Web front end. Compare two players instantly, or ask a free-text question
and get an argued verdict.

Both call into the same backends the CLI uses (tools.compare/get_player,
debate.Debate) instead of reimplementing anything, so this page and the
terminal can't drift apart. No key configured -> the debate falls back to
the offline mock provider; the comparison needs no AI at all.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Bridge Streamlit's secrets store into the environment gateway.py already
# reads from -- gateway.py is shared with the CLI and deliberately knows
# nothing about Streamlit, so this is the one place that translates.
for _key in ("GOOGLE_API_KEY", "GOOGLE_AI_STUDIO", "GEMINI_API_KEY"):
    try:
        _val = st.secrets.get(_key)
    except Exception:
        _val = None
    if _val and not os.environ.get(_key):
        os.environ[_key] = _val

st.set_page_config(page_title="Howzat", page_icon="🏏", layout="centered")

# Same palette as build_site.py and verdict_card.py -- one design system,
# not three. No webfonts, same rule the static site holds itself to.
st.markdown(
    """
<style>
:root{
  --paper:#f4f0e6;--panel:#fbf9f3;--ink:#14120c;--muted:#6e6455;--line:#ded6c2;--hair:#e8e2d3;
  --accent-ink:#8a6b00;--accent-wash:#faf0cc;--bar:#b9ad93;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,ui-serif,serif;
  --sans:ui-sans-serif,-apple-system,"Segoe UI",Inter,system-ui,sans-serif;
}
@media(prefers-color-scheme:dark){:root{
  --paper:#14130f;--panel:#1c1a15;--ink:#f0ebdd;--muted:#9a9284;--line:#2e2b23;--hair:#26241d;
  --accent-ink:#e8be2a;--accent-wash:#332b12;--bar:#3b372d;}}

.stApp, [data-testid="stAppViewContainer"] { background:var(--paper) !important; }
[data-testid="stHeader"] { background:transparent !important; }
[data-testid="stMainBlockContainer"] { max-width:760px; padding-top:3rem; }

h1, h3 { font-family:var(--serif) !important; font-weight:400 !important;
     letter-spacing:-.02em !important; color:var(--ink) !important; }
[data-testid="stCaptionContainer"] { font-family:var(--sans) !important;
     color:var(--muted) !important; }
p, li, label, .stMarkdown { font-family:var(--serif); color:var(--ink); }

[data-testid="stTextInputField"], [data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  font-family:var(--serif) !important; background:var(--panel) !important;
  border:1px solid var(--line) !important; color:var(--ink) !important;
}
[data-testid="stTextInputField"]::placeholder { color:var(--muted) !important; opacity:1 !important; }

[data-testid="stBaseButton-primary"] {
  background:var(--accent-wash) !important; color:var(--accent-ink) !important;
  border:1px solid var(--accent-ink) !important; font-family:var(--sans) !important;
  font-weight:600 !important;
}
[data-testid="stBaseButton-primary"]:disabled {
  opacity:.45 !important; border-color:var(--line) !important; color:var(--muted) !important;
}
[data-testid="stAlertContainer"] { font-family:var(--sans) !important; }
hr { border-color:var(--line) !important; }

/* ---- stats row, compare card: same classes/behavior as build_site.py's
   own compare tool, so this reads as the same product. ---------------- */
.stats{display:flex;gap:0;border-top:1px solid var(--line);border-bottom:3px double var(--line);
  margin:22px 0 30px;flex-wrap:wrap}
.stat{padding:14px 22px 13px 0;margin-right:22px;border-left:1px solid var(--hair)}
.stat:first-child{border-left:0;padding-left:0}
.stat b{display:block;font:600 26px var(--sans);font-variant-numeric:tabular-nums;
  letter-spacing:-.02em;margin-bottom:5px;color:var(--ink)}
.stat span{color:var(--muted);font:600 10px var(--sans);text-transform:uppercase;letter-spacing:.13em}

.vs-label{color:var(--muted);font:italic 17px var(--serif);text-align:center;padding-top:2.1rem}
.verdict{padding:18px 0 6px}
.vhead{font:400 27px/1.2 var(--serif);letter-spacing:-.015em;margin:0 0 6px;color:var(--ink)}
.vsub{color:var(--muted);font-size:15px;margin:0 0 18px;max-width:44em;font-family:var(--serif)}
.vrow{display:flex;gap:16px;flex-wrap:wrap}
.vcard{flex:1;min-width:200px;padding:14px 16px 15px;border-left:3px solid var(--line)}
.vcard b{display:block;font:600 15px var(--sans);margin-bottom:6px;letter-spacing:.01em;color:var(--ink)}
.vcard .big{font:600 44px/1 var(--sans);font-variant-numeric:tabular-nums;letter-spacing:-.03em;color:var(--ink)}
.vcard small{color:var(--muted);font:12.5px/1.5 var(--sans);display:block;margin-top:8px}
.win{border-left-color:var(--accent-ink)}
.win .big{color:var(--accent-ink)}
.tie{padding:13px 16px;border-left:3px solid var(--accent-ink);background:var(--accent-wash);
  font-size:14.5px;margin-top:16px;border-radius:0 3px 3px 0;font-family:var(--sans);color:var(--ink)}
.cmp{margin-top:18px;padding-top:16px;border-top:1px solid var(--hair)}
.cmprow{display:grid;grid-template-columns:minmax(100px,auto) 1fr;gap:14px;align-items:center;margin-bottom:11px}
.cmprow span{font:600 12px var(--sans);color:var(--muted);text-align:right;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.cmptrack{position:relative;height:22px}
.cmptrack::before{content:"";position:absolute;left:0;right:0;top:50%;height:1px;background:var(--hair)}
.cmpband{position:absolute;top:6px;height:10px;background:var(--bar);opacity:.6;border-radius:2px}
.cmpband.win{background:var(--accent-ink);opacity:.85}
.cmptick{position:absolute;top:2px;width:2px;height:18px;background:var(--ink)}
</style>
""",
    unsafe_allow_html=True,
)

import tools as T  # noqa: E402

FORMAT_LABELS = {
    "test": "Test batting", "odi": "ODI batting", "t20i": "T20I batting",
    "wtest": "Women's Test batting", "wodi": "Women's ODI batting", "wt20i": "Women's T20I batting",
}


@st.cache_data(ttl=3600)
def _player_names(fmt: str) -> list[str]:
    return sorted(T._ratings(fmt).player.tolist())


@st.cache_data(ttl=3600)
def _site_stats() -> dict:
    r = T._ratings("test")
    return {
        "players": int(len(r)),
        "span": f"{int(r.first_year.min())}–{int(r.last_year.max())}",
        "innings": int(r.innings.sum()),
    }


@st.cache_resource
def _provider():
    """Live gateway if the key actually works, else the offline mock --
    mirrors howzat.py's own _provider() so both entry points behave the
    same way when no key is configured."""
    from mock_provider import MockProvider
    try:
        from gateway import Gateway
        gw = Gateway()
        gw.list_models()  # cheapest possible auth probe
        return gw, "gemini"
    except Exception as e:
        return MockProvider(), f"mock (gateway unavailable: {str(e).splitlines()[0][:90]})"


# ---------------------------------------------------------------- masthead --
# Same masthead and subhead as the static site's homepage (build_site.py) --
# one identity, not a differently-branded second page.
st.markdown(
    """
<p style="font:600 11px var(--sans);text-transform:uppercase;letter-spacing:.18em;
   color:var(--muted);margin:0 0 18px">
  <b style="color:var(--ink);font-weight:700">Howzat</b> &nbsp;Era-adjusted cricket ratings
</p>
<h1 style="font:400 56px/0.95 var(--serif) !important;margin:0 0 16px;
   letter-spacing:-.02em">Settle <em style="font-style:italic;color:var(--accent-ink)">it.</em></h1>
<p style="font-size:17px;line-height:1.55;max-width:36em;margin:0;color:var(--ink)">
  Pick two players. You get a verdict and the number behind it. When the data
  cannot separate them it says so. Every comparison is era-adjusted, so 1930
  and 2026 sit on one scale.</p>
""",
    unsafe_allow_html=True,
)

stats = _site_stats()
st.markdown(
    f"""
<div class="stats">
  <div class="stat"><b>{stats['innings']:,}</b><span>Test innings analysed</span></div>
  <div class="stat"><b>{stats['span']}</b><span>every Test ever</span></div>
  <div class="stat"><b>{stats['players']:,}</b><span>Test players</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------- compare tool --
fmt = st.selectbox("Format", list(FORMAT_LABELS), format_func=lambda k: FORMAT_LABELS[k])
names = _player_names(fmt)
_default_a = names.index("DG Bradman") if fmt == "test" and "DG Bradman" in names else 0
_default_b = names.index("SR Tendulkar") if fmt == "test" and "SR Tendulkar" in names else min(1, len(names) - 1)

col_a, col_vs, col_b = st.columns([5, 1, 5])
with col_a:
    player_a = st.selectbox("Player A", names, index=_default_a)
with col_vs:
    st.markdown('<div class="vs-label">vs</div>', unsafe_allow_html=True)
with col_b:
    player_b = st.selectbox("Player B", names, index=_default_b)

if player_a == player_b:
    st.info("Pick two different players to compare.")
else:
    result = T.compare([player_a, player_b], format=fmt)
    if "error" in result:
        st.warning(result["error"])
    else:
        p = {row["player"]: row for row in result["players"]}
        pa, pb = p[player_a], p[player_b]
        leader = result["leader"]
        separated = result["intervals_separated"]

        lo = min(pa["cri_plus_95_low"], pb["cri_plus_95_low"])
        hi = max(pa["cri_plus_95_high"], pb["cri_plus_95_high"])
        span_pad = (hi - lo) * 0.08 or 1
        lo, hi = lo - span_pad, hi + span_pad

        def _band(row: dict, is_leader: bool) -> str:
            left = (row["cri_plus_95_low"] - lo) / (hi - lo) * 100
            width = (row["cri_plus_95_high"] - row["cri_plus_95_low"]) / (hi - lo) * 100
            tick = (row["rating"] - lo) / (hi - lo) * 100
            cls = "cmpband win" if is_leader else "cmpband"
            return (
                f'<div class="cmprow"><span>{row["player"]}</span><div class="cmptrack">'
                f'<div class="{cls}" style="left:{left:.1f}%;width:{width:.1f}%"></div>'
                f'<div class="cmptick" style="left:{tick:.1f}%"></div></div></div>'
            )

        verdict_line = f"{leader}. Not close." if separated else f"{leader} leads, but not decisively."

        st.markdown(
            f"""
<div class="verdict">
  <p class="vhead">{verdict_line}</p>
  <p class="vsub">{result['separation_note']}</p>
  {'<div class="tie">The top two 95% intervals overlap -- the data cannot separate them confidently.</div>' if not separated else ''}
  <div class="vrow">
    <div class="vcard {'win' if pa['player'] == leader else ''}">
      <b>{pa['player']} <span style="color:var(--muted);font-weight:400">{pa['country']}</span></b>
      <span class="big">{pa['rating']:.0f}</span>
      <small>95% interval {pa['cri_plus_95_low']:.0f}–{pa['cri_plus_95_high']:.0f} &middot;
        {pa['innings']} inns &middot; avg {pa['average']:.1f} &middot; rank #{pa['cri_plus_rank']}</small>
    </div>
    <div class="vcard {'win' if pb['player'] == leader else ''}">
      <b>{pb['player']} <span style="color:var(--muted);font-weight:400">{pb['country']}</span></b>
      <span class="big">{pb['rating']:.0f}</span>
      <small>95% interval {pb['cri_plus_95_low']:.0f}–{pb['cri_plus_95_high']:.0f} &middot;
        {pb['innings']} inns &middot; avg {pb['average']:.1f} &middot; rank #{pb['cri_plus_rank']}</small>
    </div>
  </div>
  <div class="cmp">
    {_band(pa, pa['player'] == leader)}
    {_band(pb, pb['player'] == leader)}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

st.divider()

# ------------------------------------------------------------- ask/debate --
st.markdown("### Or ask it anything")
st.caption(
    "Four agents argue it out -- a statistician, a historian, a skeptic, "
    "a judge -- and every number they cite has to come from a real lookup, "
    "not a guess. Takes a couple of minutes; you'll see who's talking below."
)

provider, which = _provider()
if which != "gemini":
    st.info(f"No live key right now ({which}), so this is running on the offline demo data.")

question = st.text_input(
    "Ask Howzat",
    placeholder='e.g. "Kohli or Sachin in ODIs?" or "Best T20I bowler of all time?"',
)
ask = st.button("Settle it", type="primary", disabled=not question.strip())

if ask and question.strip():
    from debate import Debate
    from verdict_card import render_card

    STAGE_LABEL = {
        "statistician": "Statistician gathering the decisive evidence...",
        "historian": "Historian adding the era context...",
        "skeptic": "Skeptic attacking the emerging consensus...",
        "judge": "Judge weighing the panel and delivering a verdict...",
    }
    with st.status("Starting the panel...", expanded=True) as status:
        def _on_turn(role: str) -> None:
            status.update(label=STAGE_LABEL.get(role, role))
            st.write(STAGE_LABEL.get(role, role))

        transcript = Debate(provider).run(question.strip(), on_turn=_on_turn)
        status.update(label="Verdict ready.", state="complete")

    st.components.v1.html(render_card(transcript), height=900, scrolling=True)

st.divider()
st.caption(
    "Full leaderboards and every rating table: "
    "[polestarrr.github.io/howzat](https://polestarrr.github.io/howzat)"
)
