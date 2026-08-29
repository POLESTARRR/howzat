"""RAR — runs above replacement. Cricket's missing common currency.

CRI+ and BOWL+ are *rate* statistics: they say how good a player was per
innings, not how much they were worth in total. That leaves two things
unanswerable. A brilliant 30-Test career cannot be weighed against a merely
very good 150-Test one, and a batter cannot be compared to a bowler at all,
because runs scored and wickets taken are different units.

Baseball solved this with WAR. Cricket has no equivalent. RAR is the same idea
with the honest part of it kept and the speculative part left out.

Method
------
**Replacement level** is what a team could get for free — the fringe player
who fills in. It is set at the 25th percentile of qualified players, computed
*per format and era* so it moves with the game rather than being a fixed
number.

**Batting RAR** = (player runs per dismissal − replacement) × dismissals faced.
Runs produced beyond what a replacement would have produced with the same
opportunities.

**Bowling RAR** = (replacement runs per wicket − player runs per wicket) ×
wickets. Runs *prevented* beyond replacement. Same unit, so the two add.

Both are era-adjusted through the same offsets the ratings use, so a run in
1890 and a run in 2020 count the same.

What this deliberately does NOT do
----------------------------------
It stops at runs and does not convert to wins. That conversion needs match
outcomes — which innings actually produced victories — and this dataset is
innings-level with no results column. Inventing a runs-per-win constant would
make the number look more finished and be less true, so it is left out until
the match data exists to estimate it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

REPLACEMENT_PCTILE = 25
MIN_INNINGS = {"test": 20, "odi": 25, "t20i": 20, "wodi": 25, "wt20i": 20, "wtest": 8}
MIN_BALLS = {"test": 3000, "odi": 1500, "t20i": 600,
             "wodi": 1200, "wt20i": 500, "wtest": 1000}


def _era_replacement(df: pd.DataFrame, min_inns: int) -> pd.Series:
    """Replacement runs-per-dismissal per decade, from qualified players only."""
    out = {}
    for dec, g in df.groupby("decade"):
        per = g.groupby("player").apply(
            lambda d: d.runs.sum() / max(len(d) - int(d.not_out.sum()), 1),
            include_groups=False,
        )
        counts = g.groupby("player").size()
        per = per[counts >= max(min_inns // 4, 5)]
        out[dec] = float(np.percentile(per, REPLACEMENT_PCTILE)) if len(per) >= 8 else np.nan
    s = pd.Series(out)
    return s.fillna(s.median())


def batting_rar(fmt: str = "test") -> pd.DataFrame:
    from formats import load_format

    df = load_format(fmt)
    repl = _era_replacement(df, MIN_INNINGS.get(fmt, 20))
    df = df.assign(repl=df["decade"].map(repl))

    g = df.groupby("player")
    out = pd.DataFrame({
        "country": g.country.agg(lambda s: s.mode().iat[0]),
        "innings": g.size(),
        "runs": g.runs.sum(),
        "not_outs": g.not_out.sum(),
        # Expected runs from a replacement player given the same dismissals,
        # summed innings by innings so each is charged its own era's baseline.
        "repl_runs": df.assign(r=df.repl * (~df.not_out)).groupby("player").r.sum(),
        "first_year": g.year.min(), "last_year": g.year.max(),
    })
    out["dismissals"] = (out.innings - out.not_outs).clip(lower=1)
    out["average"] = out.runs / out.dismissals
    out["bat_rar"] = out.runs - out.repl_runs
    return out.reset_index()


def bowling_rar(fmt: str = "test") -> pd.DataFrame:
    from bowl_plus import load_bowling

    df = load_bowling(fmt)
    g = df.groupby("decade")
    era_avg = (g.runs_conceded.sum() / g.wickets.sum().clip(lower=1))

    per = df.groupby(["decade", "player"]).agg(
        r=("runs_conceded", "sum"), w=("wickets", "sum"))
    per = per[per.w >= 20]
    repl = per.assign(avg=per.r / per.w.clip(lower=1)).groupby("decade").avg.agg(
        lambda s: float(np.percentile(s, 100 - REPLACEMENT_PCTILE)) if len(s) >= 8 else np.nan)
    repl = repl.reindex(era_avg.index)
    repl = repl.fillna(repl.median())

    df = df.assign(repl=df["decade"].map(repl))
    g2 = df.groupby("player")
    out = pd.DataFrame({
        "country": g2.country.agg(lambda s: s.mode().iat[0]),
        "wickets": g2.wickets.sum(),
        "runs_conceded": g2.runs_conceded.sum(),
        "balls": g2.balls.sum(),
        # What a replacement would have conceded taking the same wickets.
        "repl_runs": df.assign(x=df.repl * df.wickets).groupby("player").x.sum(),
    })
    out["bowl_average"] = out.runs_conceded / out.wickets.clip(lower=1)
    out["bowl_rar"] = out.repl_runs - out.runs_conceded
    return out.reset_index()


def build(fmt: str = "test") -> pd.DataFrame:
    bat = batting_rar(fmt)
    bowl = bowling_rar(fmt)
    d = bat.merge(bowl[["player", "wickets", "balls", "bowl_average", "bowl_rar"]],
                  on="player", how="outer")
    for c in ("bat_rar", "bowl_rar"):
        d[c] = d[c].fillna(0.0)
    d["total_rar"] = d.bat_rar + d.bowl_rar
    d = d[(d.innings.fillna(0) >= MIN_INNINGS.get(fmt, 20))
          | (d.balls.fillna(0) >= MIN_BALLS.get(fmt, 3000))]
    d["format"] = fmt
    d = d.sort_values("total_rar", ascending=False).reset_index(drop=True)
    d.to_parquet(PROC / f"rar_{fmt}.parquet", index=False)
    return d


if __name__ == "__main__":
    import sys

    fmt = sys.argv[1] if len(sys.argv) > 1 else "test"
    d = build(fmt)
    print(f"{fmt.upper()}: {len(d)} players with runs above replacement\n")
    cols = ["player", "country", "innings", "average", "bat_rar",
            "wickets", "bowl_average", "bowl_rar", "total_rar"]
    show = d.head(20)[cols].copy()
    print(show.to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    print("\nBEST BY BOWLING ALONE")
    print(d.nlargest(6, "bowl_rar")[["player", "country", "wickets",
          "bowl_average", "bowl_rar"]].to_string(index=False,
          float_format=lambda v: f"{v:.0f}"))
