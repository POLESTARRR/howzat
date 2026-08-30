"""ALL+ : an era-adjusted all-rounder rating.

The usual test — "batting average higher than bowling average" — is a blunt
instrument. It ignores era entirely (a bowling average of 25 meant something
very different in 1890 and 2005), and it treats a marginal contribution in one
discipline as equal to dominance in the other.

ALL+ combines the two ratings that already handle era properly:

    ALL+ = harmonic_mean(CRI+, BOWL+)

A harmonic mean is the right shape here. It is dominated by the *weaker* of the
two, so a great batter who bowls occasionally scores near their bowling number,
not near their batting one. That is the correct behaviour: an all-rounder is
someone with no weak side, not someone brilliant at one thing who also turns
an arm over.

Qualification is deliberately strict, because the interesting failure mode is
specialists leaking in: a batter needs enough innings AND a bowler enough
balls before they are considered an all-rounder at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# Workload thresholds scale with the format. A T20I bowler delivers 24 balls a
# match, so Test-sized bars would qualify nobody.
QUAL = {
    #        inns  balls  wkts  max balls/wkt
    "test": (30,   3000,  75,   85),
    "odi":  (40,   1500,  50,   60),
    "t20i": (30,   600,   25,   30),
    # Women's cricket is played far less; bars scale to the volume available.
    "wtest": (8,    600,   20,   90),
    "wodi": (25,   1000,   30,   70),
    "wt20i": (20,   400,   15,   35),
}
MIN_INNINGS, MIN_BALLS, MIN_WICKETS, MAX_BALLS_PER_WICKET = QUAL["test"]

# Balls alone is not enough. Allan Border accumulated 3,185 balls across a
# 265-innings career and took 33 wickets at 38 -- a batsman who turned an arm
# over, not an all-rounder. Requiring both a wicket count and a strike rate
# keeps part-timers out: Imran struck every 49 balls, Kapil every 65, Border
# every 97.


def build(fmt: str = "test") -> pd.DataFrame:
    bat_path = PROC / f"cri_plus_{fmt}.parquet"
    bowl_path = PROC / f"bowl_plus_{fmt}.parquet"
    for p in (bat_path, bowl_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — build batting and bowling first")

    bat = pd.read_parquet(bat_path)
    bowl = pd.read_parquet(bowl_path)

    bat_metric = "bat_plus" if "bat_plus" in bat.columns else "cri_plus"
    b = bat[["player", "country", "innings", "average", bat_metric]].rename(
        columns={bat_metric: "bat_rating", "average": "bat_average"}
    )
    w = bowl[["player", "wickets", "balls", "bowl_average", "bowl_plus"]]

    d = b.merge(w, on="player", how="inner")
    min_inns, min_balls, min_wkts, max_bpw = QUAL.get(fmt, QUAL["test"])
    d = d[(d.innings >= min_inns)
          & (d.balls >= min_balls)
          & (d.wickets >= min_wkts)
          & (d.balls / d.wickets.clip(lower=1) <= max_bpw)].copy()

    x, y = d.bat_rating.clip(lower=1), d.bowl_plus.clip(lower=1)
    d["all_plus"] = 2 * x * y / (x + y)
    # How lopsided they are: 0 = perfectly balanced, 1 = entirely one-sided.
    d["imbalance"] = (x - y).abs() / (x + y)

    d["format"] = fmt
    d = d.sort_values("all_plus", ascending=False).reset_index(drop=True)
    d.to_parquet(PROC / f"all_plus_{fmt}.parquet", index=False)
    return d


if __name__ == "__main__":
    import sys

    fmt = sys.argv[1] if len(sys.argv) > 1 else "test"
    d = build(fmt)
    cols = ["player", "country", "innings", "bat_average", "bat_rating",
            "wickets", "bowl_average", "bowl_plus", "all_plus"]
    q = QUAL.get(fmt, QUAL["test"])
    print(f"{fmt.upper()}: {len(d)} qualified all-rounders "
          f"(>= {q[0]} inns, {q[1]} balls, {q[2]} wkts)\n")
    print(d.head(20)[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\nMOST BALANCED (lowest imbalance, min ALL+ 100)")
    bal = d[d.all_plus >= 100].nsmallest(8, "imbalance")
    print(bal[["player", "country", "bat_rating", "bowl_plus", "all_plus",
               "imbalance"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
