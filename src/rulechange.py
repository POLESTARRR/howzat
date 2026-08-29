"""What each rule change actually did to scoring, with 95% intervals.

Cricket's Laws and playing conditions have changed constantly, and the effects
are usually asserted rather than measured ("two new balls killed reverse
swing", "covered pitches helped batters"). This measures them.

Method
------
A naive before/after mean is not enough, because scoring drifts for reasons
that have nothing to do with the rule. So each change is estimated two ways
and both are reported:

1. **Before/after** over a symmetric window, with a bootstrap 95% interval on
   the difference in runs per dismissal.
2. **Interrupted time series**: fit a linear trend on the pre-period, project
   it forward, and measure how far the post-period departs from that
   projection. If a change merely continued an existing trend, this catches it
   where a raw difference would not.

Both are reported with 95% confidence intervals, and a permutation test gives
a p-value against the null that the change date is arbitrary. A result is only
called significant when the interval excludes zero AND the permutation test
agrees, which guards against reading noise as an effect.

Limitations, stated rather than buried: these are observational, so the
estimates are associations. Several changes cluster in time and cannot be
cleanly separated; those are flagged `confounded`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

RNG = np.random.default_rng(20260830)
N_BOOT = 4000
N_PERM = 4000


@dataclass
class RuleChange:
    key: str
    year: int          # first season under the new rule
    formats: tuple[str, ...]
    title: str
    expected: str      # what the game believes happened
    confounded: bool = False


# Dated from the Laws of Cricket and ICC playing conditions.
CHANGES: list[RuleChange] = [
    RuleChange("lbw_1935", 1935, ("test",),
               "LBW law widened to balls pitching outside off stump",
               "should favour bowlers"),
    RuleChange("covered_pitches", 1972, ("test",),
               "Pitches routinely covered against rain",
               "should favour batters: no more drying sticky wickets"),
    RuleChange("front_foot_noball", 1969, ("test",),
               "Front-foot no-ball law replaces the back-foot rule",
               "modest, bowlers lose a little", confounded=True),
    RuleChange("bouncer_limit", 1991, ("test",),
               "Bouncers restricted (one, later two, per over)",
               "should favour batters"),
    RuleChange("helmets", 1978, ("test",),
               "Helmets adopted widely",
               "should favour batters", confounded=True),
    RuleChange("odi_powerplay", 2005, ("odi",),
               "Powerplay overs introduced",
               "should raise scoring"),
    RuleChange("odi_two_balls", 2011, ("odi",),
               "Two new balls, one from each end",
               "should raise scoring: no old-ball reverse swing"),
    RuleChange("odi_four_out", 2012, ("odi",),
               "Only four fielders allowed outside the circle",
               "should raise scoring"),
    RuleChange("odi_five_out", 2015, ("odi",),
               "Five fielders allowed outside in the final overs",
               "should lower death-overs scoring"),
    RuleChange("t20_free_hit", 2007, ("t20i",),
               "Free hit after a front-foot no-ball",
               "small increase", confounded=True),
]


def _runs_per_dismissal(d: pd.DataFrame) -> float:
    dismissals = max(len(d) - int(d.not_out.sum()), 1)
    return float(d.runs.sum()) / dismissals


def _bootstrap_diff(pre: pd.DataFrame, post: pd.DataFrame, n: int = N_BOOT):
    """95% interval on the change in runs per dismissal, by resampling innings."""
    a = pre[["runs", "not_out"]].to_numpy()
    b = post[["runs", "not_out"]].to_numpy()

    def rpd(x):
        dis = max(len(x) - int(x[:, 1].sum()), 1)
        return x[:, 0].sum() / dis

    diffs = np.empty(n)
    for i in range(n):
        ia = RNG.integers(0, len(a), len(a))
        ib = RNG.integers(0, len(b), len(b))
        diffs[i] = rpd(b[ib]) - rpd(a[ia])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _permutation_p(pre: pd.DataFrame, post: pd.DataFrame, n: int = N_PERM) -> float:
    """Null: the split date carries no information. Shuffle labels and compare."""
    obs = abs(_runs_per_dismissal(post) - _runs_per_dismissal(pre))
    both = pd.concat([pre, post], ignore_index=True)
    k = len(pre)
    arr = both[["runs", "not_out"]].to_numpy()

    def rpd(x):
        dis = max(len(x) - int(x[:, 1].sum()), 1)
        return x[:, 0].sum() / dis

    hits = 0
    for _ in range(n):
        idx = RNG.permutation(len(arr))
        if abs(rpd(arr[idx[k:]]) - rpd(arr[idx[:k]])) >= obs:
            hits += 1
    return (hits + 1) / (n + 1)


def _its(df: pd.DataFrame, year: int, window: int):
    """Interrupted time series: does the post-period leave the pre-trend?"""
    yearly = (df.groupby("year")
                .apply(_runs_per_dismissal, include_groups=False)
                .rename("rpd").reset_index())
    pre = yearly[(yearly.year >= year - window) & (yearly.year < year)]
    post = yearly[(yearly.year >= year) & (yearly.year < year + window)]
    if len(pre) < 4 or len(post) < 4:
        return None
    slope, intercept = np.polyfit(pre.year, pre.rpd, 1)
    projected = intercept + slope * post.year
    resid = pre.rpd - (intercept + slope * pre.year)
    sd = float(resid.std(ddof=2)) or 1e-9
    departure = float((post.rpd - projected).mean())
    se = sd / np.sqrt(len(post))
    return departure, departure - 1.96 * se, departure + 1.96 * se


def analyse(window: int = 8) -> pd.DataFrame:
    from cri_plus import canonicalise

    rows = []
    for ch in CHANGES:
        for fmt in ch.formats:
            path = PROC / f"{fmt}_innings.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path).dropna(subset=["runs", "year"])
            df = canonicalise(df)
            pre = df[(df.year >= ch.year - window) & (df.year < ch.year)]
            post = df[(df.year >= ch.year) & (df.year < ch.year + window)]
            if len(pre) < 400 or len(post) < 400:
                continue

            a, b = _runs_per_dismissal(pre), _runs_per_dismissal(post)
            lo, hi = _bootstrap_diff(pre, post)
            p = _permutation_p(pre, post)
            its = _its(df, ch.year, window)

            rows.append({
                "change": ch.key, "format": fmt, "year": ch.year,
                "title": ch.title, "expected": ch.expected,
                "pre_avg": round(a, 2), "post_avg": round(b, 2),
                "diff": round(b - a, 2), "ci_low": round(lo, 2), "ci_high": round(hi, 2),
                "p_value": round(p, 4),
                "its_effect": round(its[0], 2) if its else None,
                "its_low": round(its[1], 2) if its else None,
                "its_high": round(its[2], 2) if its else None,
                "significant": bool((lo > 0 or hi < 0) and p < 0.05),
                "confounded": ch.confounded,
                "pre_innings": len(pre), "post_innings": len(post),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_parquet(PROC / "rule_changes.parquet", index=False)
    return out


if __name__ == "__main__":
    out = analyse()
    if out.empty:
        print("no rule changes had enough data")
        raise SystemExit(0)
    for _, r in out.iterrows():
        verdict = "SIGNIFICANT" if r.significant else "not significant"
        flag = "  [confounded with other changes]" if r.confounded else ""
        print(f"\n{r.year}  {r.title}  [{r['format'].upper()}]")
        print(f"   expected: {r.expected}")
        print(f"   runs per dismissal {r.pre_avg} -> {r.post_avg}  "
              f"({r['diff']:+.2f}, 95% CI [{r.ci_low:+.2f}, {r.ci_high:+.2f}])")
        if r.its_effect is not None:
            print(f"   vs pre-existing trend: {r.its_effect:+.2f} "
                  f"(95% CI [{r.its_low:+.2f}, {r.its_high:+.2f}])")
        print(f"   permutation p={r.p_value:.4f}  ->  {verdict}{flag}")
    print(f"\n{int(out.significant.sum())}/{len(out)} changes show a significant effect")
