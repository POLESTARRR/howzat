#!/usr/bin/env python3
"""Howzat CLI.

    ./howzat.py rate                     refit CRI+ and write ratings
    ./howzat.py player Sachin            look up one batter
    ./howzat.py top --n 25               leaderboard
    ./howzat.py ask "Bradman or Sachin?" run a debate
    ./howzat.py site                     build the static explorer
    ./howzat.py check                    run tests + validation

`ask` uses the live gateway when GOOGLE_API_KEY works, and falls back to the
offline mock provider otherwise, so the command is always runnable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _provider(force_mock: bool = False):
    """Live gateway if the key actually works, else the mock."""
    from mock_provider import MockProvider

    if force_mock:
        return MockProvider(), "mock (forced)"
    try:
        from gateway import Gateway

        gw = Gateway()
        gw.list_models()  # cheapest possible auth probe
        return gw, "gemini"
    except Exception as e:
        msg = str(e).split("\n")[0][:90]
        return MockProvider(), f"mock (gateway unavailable: {msg})"


def cmd_rate(a) -> None:
    from cri_plus import build

    df, model, out = build(min_innings=a.min_innings)
    out.to_parquet(ROOT / "data/processed/cri_plus.parquet", index=False)
    print(f"{len(df):,} innings -> {len(out):,} rated players")
    print(f"empirical-Bayes lambda = {model.ridge_player:.3f}")
    print(f"top: {out.iloc[0].player} ({out.iloc[0].cri_plus:.0f})")


def cmd_player(a) -> None:
    import tools as T

    p = T.get_player(" ".join(a.name), format=a.format)
    if "error" in p:
        print(p["error"])
        for s in p.get("suggestions", [])[:5]:
            print(f"   did you mean {s['player']}?")
        return
    print(f"\n{p['player']}  ({p['country']}, {p['span']})  [{p['format']}]")
    print(f"  {p['innings']} inns, {p['runs']:,} runs, avg {p['average']}")
    metric = p.get("rating_metric", "cri_plus").upper().replace("_PLUS", "+")
    line = f"  {metric} {p['rating']}  (rank {p['rating_rank']})"
    if p.get("sr_plus") is not None:
        line += f"   SR+ {p['sr_plus']}"
    print(line)
    print(f"  {p['hundreds']} hundreds, {p['fifties']} fifties, HS {p['highest_score']}")
    print("\n  by opposition:")
    for o in p["by_opposition"][:5]:
        print(f"    {o['opposition']:<16} {int(o['innings']):>3} inns  avg {o['average']}")


def cmd_bowler(a) -> None:
    import tools as T

    b = T.get_bowler(" ".join(a.name), format=a.format)
    if "error" in b:
        print(b["error"])
        return
    print(f"\n{b['player']}  ({b['country']}, {b['span']})")
    print(f"  {b['wickets']} wickets, avg {b['bowling_average']}, "
          f"econ {b['economy']}, SR {b['strike_rate']}")
    print(f"  BOWL+ {b['bowl_plus']}  (rank {b['bowl_plus_rank']})")


def cmd_bowlers(a) -> None:
    import tools as T

    rows = T.bowling_leaderboard(a.n, format=a.format)
    if isinstance(rows, dict):
        print(rows["error"])
        return
    print(f"\n{'#':>3}  {'bowler':<22}{'ctry':<6}{'wkts':>6}{'avg':>7}{'econ':>7}{'BOWL+':>8}")
    for i, r in enumerate(rows, 1):
        print(f"{i:>3}  {r['player']:<22}{r['country']:<6}{r['wickets']:>6}"
              f"{r['bowl_average']:>7.2f}{r['economy']:>7.2f}{r['bowl_plus']:>8.0f}")


def cmd_build(a) -> None:
    """Rebuild every rating that has data, then the site."""
    import formats, build_site
    import bowl_plus

    formats.main()
    for fmt in ("test", "odi", "t20i"):
        try:
            _, _, out = bowl_plus.build(fmt)
            print(f"\n=== {fmt.upper()} BOWLING ({len(out):,} rated) ===")
            cols = ["player", "country", "wickets", "bowl_average", "economy", "bowl_plus"]
            print(out.head(10)[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        except FileNotFoundError as e:
            print(f"{fmt} bowling: skipped — {str(e).split(chr(10))[0]}")
    build_site.main()


def cmd_top(a) -> None:
    import tools as T

    rows = T.leaderboard(a.n, a.min_innings, format=a.format)
    key = "bat_plus" if rows and "bat_plus" in rows[0] else "cri_plus"
    label = "BAT+" if key == "bat_plus" else "CRI+"
    has_sr = bool(rows) and rows[0].get("sr_plus") is not None
    hdr = f"\n{'#':>3}  {'player':<22}{'ctry':<6}{'inns':>5}{'avg':>7}"
    hdr += f"{'SR+':>7}" if has_sr else ""
    print(hdr + f"{label:>8}   [{a.format.upper()}]")
    for i, r in enumerate(rows, 1):
        line = (f"{i:>3}  {r['player']:<22}{r['country']:<6}{r['innings']:>5}"
                f"{r['average']:>7.1f}")
        if has_sr:
            sr = r.get("sr_plus")
            # Balls faced is not recorded before the 1990s, so SR+ is genuinely
            # absent for older players rather than zero.
            ok = sr is not None and sr == sr
            line += f"{sr:>7.0f}" if ok else f"{'·':>7}"
        print(line + f"{r[key]:>8.0f}")


def cmd_ask(a) -> None:
    from debate import Debate
    from verdict_card import render_card

    provider, which = _provider(a.mock)
    print(f"[provider: {which}]\n")

    tr = Debate(provider).run(" ".join(a.question))
    for t in tr.turns[:-1]:
        used = f"  ({', '.join(c['name'] for c in t.tool_calls)})" if t.tool_calls else ""
        print(f"── {t.role.upper()}{used}\n   {t.text}\n")

    v = tr.verdict or {}
    print("═" * 62)
    print(f"VERDICT   {v.get('verdict')}")
    print(f"CONFIDENCE {v.get('confidence')}%")
    print(f"BECAUSE   {v.get('decisive_stat')}")
    if v.get("dissent"):
        print(f"DISSENT   {v['dissent']}")
    print("═" * 62)

    ung = tr.ungrounded_numbers()
    if ung:
        print(f"\n⚠  ungrounded numbers (no tool returned these): {', '.join(ung)}")
    else:
        print("\n✓ every number in the verdict traces to a tool result")

    if getattr(provider, "usage", None):
        print(f"   cost: {provider.usage.summary()}")

    if a.card:
        dest = ROOT / "out" / "verdict.html"
        dest.write_text(render_card(tr), encoding="utf-8")
        print(f"   card -> {dest}")


def cmd_site(a) -> None:
    import build_site

    build_site.main()


def cmd_check(a) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    t = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    print(t.stderr.strip().splitlines()[-1] if t.stderr else "")
    v = subprocess.run(
        [sys.executable, "src/validate.py"], cwd=ROOT, env=env, capture_output=True, text=True
    )
    for line in v.stdout.splitlines():
        if "checks passed" in line or "[FAIL]" in line:
            print(line)
    sys.exit(t.returncode or v.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(prog="howzat", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rate", help="refit CRI+")
    r.add_argument("--min-innings", type=int, default=20)
    r.set_defaults(fn=cmd_rate)

    p = sub.add_parser("player", help="look up a batter")
    p.add_argument("name", nargs="+")
    p.add_argument("-f", "--format", default="test", choices=("test", "odi", "t20i", "wtest", "wodi", "wt20i"))
    p.set_defaults(fn=cmd_player)

    t = sub.add_parser("top", help="leaderboard")
    t.add_argument("--n", type=int, default=20)
    t.add_argument("--min-innings", type=int, default=40)
    t.add_argument("-f", "--format", default="test", choices=("test", "odi", "t20i", "wtest", "wodi", "wt20i"))
    t.set_defaults(fn=cmd_top)

    k = sub.add_parser("ask", help="run a debate")
    k.add_argument("question", nargs="+")
    k.add_argument("--mock", action="store_true", help="force the offline provider")
    k.add_argument("--card", action="store_true", help="also write out/verdict.html")
    k.set_defaults(fn=cmd_ask)

    bw = sub.add_parser("bowler", help="look up a bowler")
    bw.add_argument("name", nargs="+")
    bw.add_argument("-f", "--format", default="test", choices=("test", "odi", "t20i", "wtest", "wodi", "wt20i"))
    bw.set_defaults(fn=cmd_bowler)

    bl = sub.add_parser("bowlers", help="bowling leaderboard")
    bl.add_argument("--n", type=int, default=20)
    bl.add_argument("-f", "--format", default="test", choices=("test", "odi", "t20i", "wtest", "wodi", "wt20i"))
    bl.set_defaults(fn=cmd_bowlers)

    bd = sub.add_parser("build", help="rebuild all ratings + site")
    bd.set_defaults(fn=cmd_build)

    s = sub.add_parser("site", help="build the static explorer")
    s.set_defaults(fn=cmd_site)

    c = sub.add_parser("check", help="tests + validation")
    c.set_defaults(fn=cmd_check)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
