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


st.title("🏏 Howzat")
st.caption(
    "Era-adjusted cricket ratings. Ask a question and get a verdict grounded "
    "in every Test/ODI/T20I innings since 1877 -- never a bare opinion."
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
