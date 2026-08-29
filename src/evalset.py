"""An evaluation set for the debate panel.

Anecdotes are not evidence. Three good transcripts prove nothing about whether
the panel abstains when it should, uses the right format, or invents numbers.
This scores those properties on questions whose correct behaviour is known in
advance and checkable in code.

Cases are graded on *behaviour*, not prose, so grading needs no second model:

  leader      the verdict must name a specific player the data supports
  abstain     the ratings genuinely cannot separate them; declaring a winner
              is a failure, saying so is a pass
  format      an ODI/T20I question must be answered from that format's data
  grounding   every number in the verdict must trace to a tool result

Run cheaply against the mock provider in CI, or live to get a real score:

    python3 src/evalset.py --mock
    python3 src/evalset.py --live --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tools as T
from debate import Debate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


@dataclass
class Case:
    id: str
    question: str
    kind: str                      # leader | abstain | format | grounding
    fmt: str = "test"
    players: tuple[str, ...] = ()
    expect_leader: str | None = None


def _separated(names, fmt) -> tuple[bool, str | None, str | None]:
    """Ground truth straight from the ratings, not from an opinion."""
    c = T.compare(list(names), format=fmt)
    if "error" in c:
        return False, None, c["error"]
    return bool(c["intervals_separated"]), c["leader"], None


def build_cases() -> list[Case]:
    """Cases are derived from the data, so they stay true as ratings change."""
    cases: list[Case] = []

    probes = [
        ("test", ("Bradman", "Tendulkar")),
        ("test", ("Bradman", "Hobbs")),
        ("test", ("Tendulkar", "Kallis")),
        ("test", ("Lara", "Ponting")),
        ("test", ("Smith", "Root")),
        ("odi", ("Viv", "Kohli")),
        ("odi", ("Kohli", "Tendulkar")),
        ("odi", ("Dhoni", "Bevan")),
        ("t20i", ("Kohli", "Buttler")),
        ("t20i", ("SA Yadav", "Salt")),
    ]
    for fmt, pair in probes:
        sep, leader, err = _separated(pair, fmt)
        if err:
            continue
        label = {"test": "Test", "odi": "ODI", "t20i": "T20I"}[fmt]
        q = f"In {label} cricket, who was the better batsman: {pair[0]} or {pair[1]}?"
        if sep:
            cases.append(Case(f"lead_{fmt}_{pair[0]}_{pair[1]}".lower(), q,
                              "leader", fmt, pair, leader))
        else:
            cases.append(Case(f"abst_{fmt}_{pair[0]}_{pair[1]}".lower(), q,
                              "abstain", fmt, pair))

    # Format discipline: the answer must come from the named format's data.
    cases += [
        Case("fmt_odi_viv", "Was Viv Richards the greatest ODI batsman?", "format", "odi",
             ("Viv",)),
        Case("fmt_t20_kohli", "Is Virat Kohli the best T20I batsman ever?", "format",
             "t20i", ("Kohli",)),
    ]
    # Grounding always applies; these are just extra pressure.
    cases += [
        Case("grd_goat", "Who is the greatest Test batsman of all time?", "grounding",
             "test"),
    ]
    return cases


@dataclass
class Result:
    case: Case
    passed: bool
    detail: str
    ungrounded: list[str] = field(default_factory=list)


def grade(case: Case, tr) -> Result:
    v = tr.verdict or {}
    text = " ".join(str(x) for x in v.values() if x).lower()
    ung = tr.ungrounded_numbers()

    if ung:
        return Result(case, False, f"invented numbers: {', '.join(ung)}", ung)

    if case.kind == "abstain":
        # Passing means admitting the data cannot separate them.
        admits = any(w in text for w in (
            "cannot", "can not", "not establish", "overlap", "inconclusive",
            "no clear", "cannot separate", "not separated", "too close",
        ))
        conf = v.get("confidence")
        low_conf = isinstance(conf, (int, float)) and conf < 60
        ok = admits or low_conf
        return Result(case, ok,
                      "admitted overlap" if ok else f"claimed a winner (conf={conf})", ung)

    if case.kind == "leader":
        want = (case.expect_leader or "").split()[-1].lower()
        ok = want in text
        return Result(case, ok,
                      f"named {case.expect_leader}" if ok else
                      f"expected {case.expect_leader}, verdict said: {text[:90]}", ung)

    if case.kind == "format":
        used = {json.loads(k[k.index("(") + 1:-1]).get("format", "test")
                for k in tr.facts if "(" in k}
        ok = case.fmt in {str(u).lower() for u in used}
        return Result(case, ok,
                      f"queried {sorted(used)}" if ok else
                      f"WRONG FORMAT: queried {sorted(used)}, needed {case.fmt}", ung)

    return Result(case, True, "no invented numbers", ung)


def run(live: bool, limit: int | None = None) -> list[Result]:
    if live:
        from gateway import Gateway
        provider = Gateway()
    else:
        from mock_provider import MockProvider
        provider = MockProvider(inject_hallucination=False)

    cases = build_cases()
    if limit:
        cases = cases[:limit]

    results = []
    d = Debate(provider)
    for i, c in enumerate(cases, 1):
        try:
            tr = d.run(c.question)
            r = grade(c, tr)
        except Exception as e:  # noqa: BLE001 - a provider failure is a case failure
            r = Result(c, False, f"{type(e).__name__}: {str(e)[:90]}")
        results.append(r)
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {i:2d}/{len(cases)} {c.kind:9s} {c.id[:38]:38s} {r.detail[:60]}",
              flush=True)

    if live and getattr(provider, "usage", None):
        print(f"\n  cost: {provider.usage.summary()}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use the real gateway")
    ap.add_argument("--mock", action="store_true", help="offline (default)")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    print(f"Howzat debate eval — {'LIVE' if a.live else 'MOCK'}\n")
    results = run(live=a.live, limit=a.limit)

    by_kind: dict[str, list[bool]] = {}
    for r in results:
        by_kind.setdefault(r.case.kind, []).append(r.passed)

    print(f"\n{'=' * 66}")
    for kind, vals in sorted(by_kind.items()):
        print(f"  {kind:10s} {sum(vals)}/{len(vals)}")
    total = sum(r.passed for r in results)
    print(f"  {'TOTAL':10s} {total}/{len(results)} = {total / max(len(results), 1):.0%}")
    print("=" * 66)

    OUT.mkdir(exist_ok=True)
    (OUT / "eval_results.json").write_text(json.dumps(
        [{"id": r.case.id, "kind": r.case.kind, "passed": r.passed,
          "detail": r.detail} for r in results], indent=2))
    sys.exit(0 if total == len(results) else 1)


if __name__ == "__main__":
    main()
