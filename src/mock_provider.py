"""An offline stand-in for the LLM gateway.

Layer 2 is blocked on a working API key, but the orchestration should not be
untested until that arrives. MockProvider speaks the same response shape as the
Gemini gateway (functionCall parts, then text), so `Debate` can be exercised end
to end: tool dispatch, evidence accumulation, role sequencing, JSON verdict
parsing and the grounding check.

It deliberately emits one *ungrounded* number in the verdict so the grounding
check has something to catch. A guard that has never fired is not a guard.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _fc(name: str, args: dict) -> dict:
    return {"candidates": [{"content": {"parts": [{"functionCall": {"name": name, "args": args}}]}}]}


def _txt(s: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": s}]}}]}


class MockProvider:
    """Scripted provider. No network, deterministic."""

    def __init__(self, inject_hallucination: bool = True):
        self.inject_hallucination = inject_hallucination
        self.log: list[tuple[str, str]] = []

    # -- shape-compatible with gateway.Gateway ---------------------------

    @staticmethod
    def text(resp: dict) -> str:
        parts = resp.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip()

    @staticmethod
    def calls(resp: dict) -> list[dict]:
        parts = resp.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return [p["functionCall"] for p in parts if "functionCall" in p]

    # -- the script ------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        tier: str = "cheap",
        system: str | None = None,
        tools: list[dict] | None = None,
        history: list[dict] | None = None,
        **_: Any,
    ) -> dict:
        role = self._role(system or "")
        seen_tools = prompt.startswith("Tool results:")
        self.log.append((role, "post-tool" if seen_tools else "initial"))

        names = _players_in(prompt)

        if role == "statistician":
            if not seen_tools:
                return _fc("compare", {"names": names or ["Bradman", "Tendulkar"], "format": "test"})
            return _txt(
                "On CRI+ the order is clear, and the intervals do not overlap at the top. "
                "Raw average and CRI+ agree here, so the era adjustment is not the "
                "deciding factor."
            )

        if role == "historian":
            if not seen_tools:
                return _fc("era_context", {"decade": 1930})
            return _txt(
                "The 1930s were the most forgiving era in Test history for batting, which "
                "is exactly what CRI+ discounts. That the leader survives the discount is "
                "the meaningful part."
            )

        if role == "skeptic":
            if not seen_tools:
                return _fc("get_player", {"name": names[0] if names else "Bradman"})
            return _txt(
                "The sample is the weak point: a short career leaves a wide interval, and "
                "an interval that wide cannot separate second from fifth. The top place is "
                "safe; every ranking below it is not."
            )

        # judge
        stat = "CRI+ 336 with a 95% interval of [269, 420]"
        dissent = "Places below first are not separated; those intervals overlap."
        if self.inject_hallucination:
            # A figure no tool returned, to prove the grounding check fires.
            dissent += " Career strike rate of 87.4 also favours the modern player."
        return _txt(
            json.dumps(
                {
                    "verdict": "Bradman, and not narrowly.",
                    "confidence": 92,
                    "decisive_stat": stat,
                    "dissent": dissent,
                }
            )
        )

    @staticmethod
    def _role(system: str) -> str:
        for r in ("statistician", "historian", "skeptic", "judge"):
            if f"the {r}".upper() in system.upper():
                return r
        return "judge"


def _players_in(prompt: str) -> list[str]:
    """Pull capitalised surnames out of the question for the mock to look up."""
    q = prompt.split("\n")[0]
    found = re.findall(r"\b[A-Z][a-z]{3,}\b", q)
    stop = {
        "The", "question", "Tool", "Panel", "Evidence", "Gather", "Deliver",
        "Attack", "Test", "Tests", "Match", "Cricket", "Batter", "Batsman",
        "Bowler", "Greatest", "Best", "Better", "Than", "Which", "Compare",
    }
    return [f for f in found if f not in stop][:3]
