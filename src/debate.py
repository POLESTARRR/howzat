"""Multi-agent debate: four agents argue a cricket question, grounded in CRI+.

Structure follows the courtroom-style debate framings used for claim
verification: specialised roles, an adversary whose job is to disagree, and a
judge that synthesises. The adversary matters. The known failure mode of
multi-agent debate is confident agreement on a wrong answer, so the Skeptic is
instructed to attack the emerging consensus rather than help build it.

Grounding is the other guard. Agents may only assert numbers that came back
from a tool call; `Transcript.ungrounded_numbers` reports any figure in the
final verdict that no tool produced, and that count is part of the output rather
than hidden.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import tools as T


class Provider(Protocol):
    """Anything that can answer a prompt. Gateway or MockProvider."""

    def generate(self, prompt: str, **kw: Any) -> dict: ...

    @staticmethod
    def text(resp: dict) -> str: ...

    @staticmethod
    def calls(resp: dict) -> list[dict]: ...


# --------------------------------------------------------------------- roles

COMMON = """You are part of a panel settling a cricket argument.

The panel has access to era-adjusted ratings covering THREE FORMATS:
  test  CRI+  runs per dismissal vs the era (1877-2026)
  odi   BAT+  durability and tempo combined (1971-2026)
  t20i  BAT+  tempo-weighted (2005-2026)
100 = an average batter of the SAME ERA AND FORMAT, so eras are comparable in a
way raw batting average is not. These measure dominance over contemporaries,
not absolute skill.

Rules you must follow:
- Work out which format the question is about and pass `format` on EVERY tool
  call. Answering an ODI question with Test ratings is a serious error.
- Call tools for every number. Never state a statistic from memory.
- If a tool did not give you a figure, do not use that figure.
- CRI+ values carry 95% intervals. Overlapping intervals mean the gap is not
  established; say so rather than declaring a winner.
- Be brief. Three or four sentences.
"""

ROLES = {
    "statistician": COMMON + """
You are the STATISTICIAN. Gather the decisive evidence with tool calls and
state plainly what the numbers show. Lead with CRI+ and its interval.
""",
    "historian": COMMON + """
You are the HISTORIAN. Supply the context the raw numbers miss: what scoring
conditions were like in the relevant eras, and who the player faced. Use
era_context. Explain what the era adjustment is actually correcting for.
""",
    "skeptic": COMMON + """
You are the SKEPTIC. Your job is to attack the emerging consensus, not to agree
with it. Find the weakest link: small sample size, wide intervals, weak
opposition, a short career, an era baseline that may be distorted. If the
evidence genuinely is decisive, say so, but only after a real attempt to break
it.
""",
    "judge": COMMON + """
You are the JUDGE. Weigh what the panel said and deliver a verdict.

Reply as strict JSON, no markdown fence:
{"verdict": "<one sentence>",
 "confidence": <0-100>,
 "decisive_stat": "<the single number that settles it>",
 "dissent": "<the strongest surviving objection, or null>"}

Set confidence below 60 when CRI+ intervals overlap.
""",
}


@dataclass
class Turn:
    role: str
    text: str
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class Transcript:
    question: str
    turns: list[Turn] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    verdict: dict | None = None

    def evidence_block(self) -> str:
        if not self.facts:
            return "(no evidence gathered yet)"
        return json.dumps(self.facts, indent=2, default=str)[:6000]

    def panel_block(self) -> str:
        return "\n\n".join(f"{t.role.upper()}: {t.text}" for t in self.turns if t.text)

    def grounded_numbers(self) -> set[float]:
        """Every number that appeared in a tool result."""
        blob = json.dumps(self.facts, default=str)
        return {float(x) for x in re.findall(r"\d+(?:\.\d+)?", blob)}

    def ungrounded_numbers(self) -> list[str]:
        """Numbers in the verdict that no tool ever returned.

        Years and small integers are ignored: they are usually rhetorical
        ("three sentences", "the 1930s") rather than claimed statistics.
        """
        if not self.verdict:
            return []
        text = " ".join(str(v) for v in self.verdict.values() if v)
        grounded = self.grounded_numbers()
        out = []
        for n in re.findall(r"\d+(?:\.\d+)?", text):
            if len(n) <= 2:
                continue
            if re.fullmatch(r"(1[89]|20)\d{2}", n):  # a year
                continue
            val = float(n)
            # An agent rounding 335.9 to "336" is honest reporting, not
            # invention, so match numerically with a small tolerance.
            if any(abs(val - g) <= max(0.5, abs(g) * 0.005) for g in grounded):
                continue
            out.append(n)
        return out


class Debate:
    def __init__(self, provider: Provider, max_tool_rounds: int = 3):
        self.p = provider
        self.max_tool_rounds = max_tool_rounds

    def _agent(self, role: str, prompt: str, tr: Transcript, tier: str = "cheap") -> Turn:
        """Run one agent, letting it call tools until it produces prose."""
        turn = Turn(role=role, text="")
        history: list[dict] = []
        current = prompt

        for _ in range(self.max_tool_rounds):
            resp = self.p.generate(
                current,
                tier=tier,
                system=ROLES[role],
                tools=T.TOOL_SCHEMAS,
                history=history,
            )
            calls = self.p.calls(resp)
            if not calls:
                turn.text = self.p.text(resp)
                return turn

            results = []
            for c in calls:
                name = c.get("name", "")
                args = c.get("args", {}) or {}
                out = T.dispatch(name, args)
                turn.tool_calls.append({"name": name, "args": args})
                key = f"{name}({json.dumps(args, sort_keys=True, default=str)})"
                tr.facts[key] = out.get("result", out)
                results.append(f"{key} -> {json.dumps(out, default=str)[:6000]}")

            history = history + [{"role": "user", "parts": [{"text": current}]}]
            current = (
                "Tool results:\n" + "\n".join(results)
                + "\n\nNow give your statement. Use only these numbers."
            )

        # Ran out of tool rounds: force a final answer.
        resp = self.p.generate(current, tier=tier, system=ROLES[role], history=history)
        turn.text = self.p.text(resp)
        return turn

    def run(self, question: str, on_turn=None) -> Transcript:
        """`on_turn`, if given, is called as on_turn(role) right BEFORE each
        agent starts (statistician/historian/skeptic/judge) -- lets a caller
        show live progress instead of one long silent wait for all four
        sequential calls. Optional and stateless: the CLI doesn't pass one."""
        tr = Transcript(question=question)

        if on_turn:
            on_turn("statistician")
        tr.turns.append(
            self._agent(
                "statistician",
                f"The question: {question}\n\nGather the decisive evidence and report it.",
                tr,
            )
        )
        if on_turn:
            on_turn("historian")
        tr.turns.append(
            self._agent(
                "historian",
                f"The question: {question}\n\nThe statistician said:\n"
                f"{tr.turns[-1].text}\n\nAdd the era context this misses.",
                tr,
            )
        )
        if on_turn:
            on_turn("skeptic")
        tr.turns.append(
            self._agent(
                "skeptic",
                f"The question: {question}\n\nThe panel so far:\n{tr.panel_block()}\n\n"
                "Attack the emerging consensus. Where is it weakest?",
                tr,
            )
        )

        # Only the judge is escalated to the strong tier. If it fails outright,
        # keep the transcript: the panel's evidence is the expensive part and
        # is still worth returning.
        if on_turn:
            on_turn("judge")
        judge_prompt = (
            f"The question: {question}\n\nPanel:\n{tr.panel_block()}\n\n"
            f"Evidence gathered:\n{tr.evidence_block()}\n\nDeliver the verdict as JSON."
        )
        for tier in ("strong", "cheap"):
            try:
                judge = self._agent("judge", judge_prompt, tr, tier=tier)
                tr.turns.append(judge)
                tr.verdict = _parse_json(judge.text)
                return tr
            except Exception as e:  # noqa: BLE001 - any provider failure
                tr.errors.append(f"judge/{tier}: {type(e).__name__}: {str(e)[:160]}")

        tr.turns.append(Turn(role="judge", text=""))
        tr.verdict = {
            "verdict": "No verdict: the judge could not be reached.",
            "confidence": None,
            "decisive_stat": None,
            "dissent": None,
        }
        return tr


def _parse_json(text: str) -> dict:
    """Judges wrap JSON in prose or fences more often than they should."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"verdict": text[:400], "confidence": None, "decisive_stat": None, "dissent": None}
