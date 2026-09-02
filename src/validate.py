"""Sanity checks for CRI+.

A rating that disagrees with settled cricket opinion is not brave, it is broken.
These checks are deliberately about cases nobody argues over, so that the metric
can be trusted on the cases people do argue over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cri_plus import PROC, build

# Batters whose greatness is not in dispute; used as a recall check.
CANON = [
    "DG Bradman", "SR Tendulkar", "BC Lara", "RT Ponting", "GS Sobers",
    "JB Hobbs", "GA Headley", "KC Sangakkara", "SPD Smith", "V Kohli",
    "IVA Richards", "WR Hammond", "L Hutton", "JH Kallis", "R Dravid",
    "AB de Villiers", "JE Root", "SR Waugh", "Younis Khan", "AD Nourse",
]


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return ok


def main() -> None:
    df, model, out = build()
    print(f"\ninnings {len(df):,} | span {df.year.min()}–{df.year.max()} | rated {len(out):,}\n")
    print(f"optimiser converged={model.result_.success} nll={model.result_.fun:,.0f}\n")

    results = []

    print("CHECK 1: Bradman")
    top = out.iloc[0]
    results.append(check("Bradman is #1", top.player == "DG Bradman", f"got {top.player} ({top.cri_plus:.0f})"))
    if len(out) > 1:
        gap = top.cri_plus / out.iloc[1].cri_plus
        results.append(check("Bradman leads by a clear margin", gap > 1.15, f"{gap:.2f}x over {out.iloc[1].player}"))

    print("\nCHECK 2: era effects are directionally sane")
    era = model.era_
    print("   " + era.round(3).to_string().replace("\n", "\n   "))
    if 1950 in era.index and 2000 in era.index:
        results.append(check("2000s scored easier than 1950s", era[2000] > era[1950],
                             f"2000s={era[2000]:+.3f} vs 1950s={era[1950]:+.3f}"))

    print("\nCHECK 3: canonical greats rank highly")
    top100 = set(out.head(100).player)
    found = [p for p in CANON if p in set(out.player)]
    hits = [p for p in found if p in top100]
    rate = len(hits) / max(len(found), 1)
    results.append(check("≥80% of canonical greats in top 100", rate >= 0.8,
                         f"{len(hits)}/{len(found)} = {rate:.0%}"))
    missing = [p for p in found if p not in top100]
    if missing:
        sub = out[out.player.isin(missing)][["player", "average", "cri_plus"]]
        print("   outside top 100:")
        print("   " + sub.to_string(index=False).replace("\n", "\n   "))

    print("\nCHECK 4: shrinkage actually bites")
    lo = out[out.innings < 30]
    hi = out[out.innings >= 100]
    if len(lo) and len(hi):
        results.append(check("low-innings players are shrunk toward the mean",
                             lo.cri_plus.std() < hi.cri_plus.std() * 1.6,
                             f"sd(<30 inns)={lo.cri_plus.std():.1f} vs sd(100+)={hi.cri_plus.std():.1f}"))

    print("\nCHECK 5: CRI+ disagrees with raw average (else it adds nothing)")
    r = np.corrcoef(out.average, out.cri_plus)[0, 1]
    results.append(check("correlated but not identical to average", 0.6 < r < 0.985, f"r={r:.3f}"))

    print("\nCHECK 6: uncertainty scales with sample size")
    out["ci_width"] = out.cri_hi - out.cri_lo
    bands = [(20, 40), (40, 80), (80, 150), (150, 10_000)]
    widths = []
    for lo, hi in bands:
        s = out[(out.innings >= lo) & (out.innings < hi)]
        if len(s):
            widths.append(s.ci_width.mean())
            print(f"   {lo}-{hi if hi < 10_000 else '+'} inns: n={len(s):4d}  mean 95% CI width={s.ci_width.mean():.1f}")
    results.append(check("CI narrows monotonically with innings",
                         all(a > b for a, b in zip(widths, widths[1:])),
                         " > ".join(f"{w:.0f}" for w in widths)))

    print("\nCHECK 7: Bradman's separation survives uncertainty")
    brad = out[out.player == "DG Bradman"]
    if len(brad):
        b = brad.iloc[0]
        rest = out[out.player != "DG Bradman"]
        beaten = (rest.cri_plus < b.cri_lo).mean()
        results.append(check("Bradman's lower bound beats most point estimates",
                             beaten > 0.98, f"clears {beaten:.1%} of the field"))

    print("\nCHECK 8: biggest era corrections")
    o = out.copy()
    o["avg_rank"] = o.average.rank(ascending=False)
    o["cri_rank"] = o.cri_plus.rank(ascending=False)
    o["move"] = o.avg_rank - o.cri_rank
    qual = o[o.innings >= 50]
    cols = ["player", "country", "innings", "average", "cri_plus", "avg_rank", "cri_rank", "move"]
    print("   biggest RISES (era-adjusted upward):")
    print("   " + qual.nlargest(8, "move")[cols].to_string(index=False).replace("\n", "\n   "))
    print("\n   biggest FALLS:")
    print("   " + qual.nsmallest(8, "move")[cols].to_string(index=False).replace("\n", "\n   "))

    print(f"\n{'='*66}")
    print(f"{sum(results)}/{len(results)} checks passed")
    print("=" * 66)

    dest = PROC / "cri_plus.parquet"
    out.to_parquet(dest, index=False)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
