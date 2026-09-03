"""Howzat, on the web: ask it a question, get a grounded verdict.

Thin UI over the existing `Debate`/`gateway` backend (same one `howzat.py
ask` uses) -- no rating logic lives here, so the CLI and this page can
never disagree. Falls back to the offline mock provider automatically if
no working Gemini key is configured, so the page never hard-fails.
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

# Same "Wisden Almanack" palette as the static site and the verdict card
# (src/build_site.py, src/verdict_card.py) -- one design system across all
# three surfaces. System-resident fonts only, no webfont/CDN request, same
# constraint the static site holds itself to. Selectors are the same
# data-testid set already confirmed against an installed Streamlit>=1.32
# build for this session's other app (Recast) -- not guessed blind.
st.markdown(
    """
<style>
:root{
  --paper:#f4f0e6;--panel:#fbf9f3;--ink:#14120c;--muted:#6e6455;--line:#ded6c2;
  --accent-ink:#8a6b00;--accent-wash:#faf0cc;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,ui-serif,serif;
  --sans:ui-sans-serif,-apple-system,"Segoe UI",Inter,system-ui,sans-serif;
}
@media(prefers-color-scheme:dark){:root{
  --paper:#14130f;--panel:#1c1a15;--ink:#f0ebdd;--muted:#9a9284;--line:#2e2b23;
  --accent-ink:#e8be2a;--accent-wash:#332b12;}}

.stApp, [data-testid="stAppViewContainer"] { background:var(--paper) !important; }
[data-testid="stHeader"] { background:transparent !important; }
[data-testid="stMainBlockContainer"] { max-width:720px; padding-top:3rem; }

h1 { font-family:var(--serif) !important; font-weight:400 !important;
     letter-spacing:-.02em !important; color:var(--ink) !important; }
[data-testid="stCaptionContainer"] { font-family:var(--sans) !important;
     color:var(--muted) !important; }
p, li, label, .stMarkdown { font-family:var(--serif); color:var(--ink); }

[data-testid="stTextInput"] input {
  font-family:var(--serif) !important; background:var(--panel) !important;
  border:1px solid var(--line) !important; color:var(--ink) !important;
}
[data-testid="baseButton-primary"] {
  background:var(--accent-wash) !important; color:var(--accent-ink) !important;
  border:1px solid var(--accent-ink) !important; font-family:var(--sans) !important;
  font-weight:600 !important;
}
[data-testid="stAlertContainer"] { font-family:var(--sans) !important; }
hr { border-color:var(--line) !important; }
</style>
""",
    unsafe_allow_html=True,
)


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


# Same masthead as the static site's homepage (build_site.py: "Howzat / Era-
# adjusted cricket ratings" eyebrow, "Settle it." headline) -- one identity,
# not a differently-branded second page.
st.markdown(
    """
<p style="font:600 11px var(--sans);text-transform:uppercase;letter-spacing:.18em;
   color:var(--muted);margin:0 0 18px">
  <b style="color:var(--ink);font-weight:700">Howzat</b> &nbsp;Era-adjusted cricket ratings
</p>
<h1 style="font:400 56px/0.95 var(--serif) !important;margin:0 0 16px;
   letter-spacing:-.02em">Settle <em style="font-style:italic;color:var(--accent-ink)">it.</em></h1>
""",
    unsafe_allow_html=True,
)
st.caption(
    "Ask a question and get a verdict grounded in every Test, ODI and T20I "
    "innings since 1877 -- never a bare opinion."
)

provider, which = _provider()
if which != "gemini":
    st.info(f"Running on the offline demo provider ({which}). Answers are illustrative, not live-computed.")

question = st.text_input(
    "Ask Howzat",
    placeholder='e.g. "Bradman or Sachin?" or "Best T20I bowler of all time?"',
)
ask = st.button("Settle it", type="primary", disabled=not question.strip())

if ask and question.strip():
    from debate import Debate
    from verdict_card import render_card

    with st.spinner("Gathering evidence and arguing it out..."):
        transcript = Debate(provider).run(question.strip())

    st.components.v1.html(render_card(transcript), height=900, scrolling=True)

st.divider()
st.caption(
    "Full leaderboards and every rating table: "
    "[polestarrr.github.io/howzat](https://polestarrr.github.io/howzat)"
)
