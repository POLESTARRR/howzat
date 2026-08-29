"""Extend CRI+ to ODI and T20I.

Tests are a longevity format: runs per dismissal is very nearly the whole story,
so CRI+ alone works. Limited-overs cricket is not like that. A batter consumes
two scarce resources, balls and wickets, and 30 off 15 can be worth more than
40 off 40. A rating that ignores tempo would rank blockers above match-winners.

So limited-overs carries two indices, both era-relative:

    CRI+  runs per dismissal vs the era      (durability)
    SR+   runs per ball vs the era           (tempo)

and a combined BAT+ that weights them by format. **The weights are a judgement
call, not a derived result**, and are stated here rather than buried:

    Test  1.00 durability / 0.00 tempo
    ODI   0.55 / 0.45
    T20I  0.30 / 0.70

BAT+ is a weighted geometric mean, so a batter has to be respectable on both
axes; being superb at one cannot fully rescue being poor at the other. Anyone
who disagrees with the weights can read CRI+ and SR+ separately, which is why
both always ship alongside the combined number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cri_plus import CriPlusModel, career_table

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

WEIGHTS = {"test": (1.00, 0.00), "odi": (0.55, 0.45), "t20i": (0.30, 0.70)}
MIN_INNINGS = {"test": 20, "odi": 25, "t20i": 20}

# ICC Full Members. Ratings only count innings against these sides.
#
# This is a scoping decision, and a necessary one. Since the ICC granted T20I
# status to every member in 2019, only 34% of T20I innings are against Full
# Members -- the other two thirds are associate sides playing each other. Left
# unfiltered the all-time T20I table fills with batters from Austria, Bulgaria
# and Spain, who score fast because those attacks are weak. No opposition
# adjustment can repair that, because associate teams mostly play *each other*,
# so the baseline is dragged down with them.
FULL_MEMBERS = {
    "Afghanistan", "Australia", "Bangladesh", "England", "India", "Ireland",
    "New Zealand", "Pakistan", "South Africa", "Sri Lanka", "West Indies",
    "Zimbabwe",
}


def load_format(fmt: str) -> pd.DataFrame:
    path = PROC / f"{fmt}_innings.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run: python3 src/fetch_statsguru.py {fmt}")
    df = pd.read_parquet(path)
    df = df.dropna(subset=["runs", "year"]).copy()
    df["decade"] = (df["year"] // 10 * 10).astype(int)
    from cri_plus import canonicalise

    df = canonicalise(df)
    before = len(df)
    df = df[df["opposition"].isin(FULL_MEMBERS)]
    if before and len(df) < before:
        print(f"  [{fmt}] kept {len(df):,}/{before:,} innings "
              f"({len(df)/before:.0%}) vs Full Member opposition")
    return df


# Balls of evidence before SR+ is trusted at face value. Below this the index is
# pulled toward 100, so 22 innings against minnows cannot top the table.
SR_PRIOR_BALLS = 600


def strike_rate_index(df: pd.DataFrame, players: pd.Index) -> pd.Series:
    """Runs per ball vs *expected*, indexed to 100.

    Expected is set by both the era and the opposition. Era alone is not
    enough: without an opposition term the T20I table filled up with batters
    from Austria, Bulgaria and Spain, who score fast because associate attacks
    are weak, not because they are the best strikers alive. CRI+ already
    adjusts for opposition; SR+ has to as well or the combined rating inherits
    the bias.

    The index is then shrunk toward 100 by balls faced, since a huge SR+ over
    300 balls is mostly noise.
    """
    d = df.dropna(subset=["balls_faced"])
    d = d[d["balls_faced"] > 0]
    if d.empty:
        return pd.Series(np.nan, index=players)

    total_rpb = d.runs.sum() / d.balls_faced.sum()

    era = d.groupby("decade").apply(
        lambda g: g.runs.sum() / g.balls_faced.sum(), include_groups=False
    )
    # How freely runs flow against this opposition, relative to all cricket.
    opp = d.groupby("opposition").apply(
        lambda g: g.runs.sum() / g.balls_faced.sum(), include_groups=False
    )
    opp_factor = (opp / total_rpb).clip(lower=0.5, upper=2.0)

    d = d.assign(
        exp_rpb=d["decade"].map(era) * d["opposition"].map(opp_factor).fillna(1.0)
    )

    g = d.groupby("player")
    runs = g.runs.sum()
    balls = g.balls_faced.sum()
    expected = d.assign(exp=d.exp_rpb * d.balls_faced).groupby("player").exp.sum()

    raw = 100 * runs / expected.replace(0, np.nan)
    # Shrink toward 100 in proportion to evidence.
    w = balls / (balls + SR_PRIOR_BALLS)
    idx = 100 + (raw - 100) * w

    idx = idx.where(g.size() >= 10)
    return idx.reindex(players)


def rate_format(fmt: str) -> pd.DataFrame:
    df = load_format(fmt)
    careers = career_table(df)
    keep = careers.loc[careers["innings"] >= MIN_INNINGS[fmt], "player"]
    sub = df[df["player"].isin(keep)].copy()

    model = CriPlusModel().fit_eb(sub)
    out = careers.merge(model.ratings(sub), on="player", how="inner")

    out["sr_plus"] = strike_rate_index(sub, pd.Index(out["player"])).to_numpy()

    wd, wt = WEIGHTS[fmt]
    if wt == 0:
        out["bat_plus"] = out["cri_plus"]
    else:
        # Weighted geometric mean; falls back to CRI+ where SR+ is unavailable.
        sr = out["sr_plus"]
        out["bat_plus"] = np.where(
            sr.notna(),
            np.power(out["cri_plus"].clip(lower=1), wd) * np.power(sr.clip(lower=1), wt),
            out["cri_plus"],
        )

    out["format"] = fmt
    out = out.sort_values("bat_plus", ascending=False).reset_index(drop=True)
    dest = PROC / f"cri_plus_{fmt}.parquet"
    out.to_parquet(dest, index=False)
    return out


def main() -> None:
    frames = []
    for fmt in ("test", "odi", "t20i"):
        try:
            out = rate_format(fmt)
        except FileNotFoundError as e:
            print(f"{fmt}: skipped — {e}")
            continue
        frames.append(out)
        wd, wt = WEIGHTS[fmt]
        print(f"\n=== {fmt.upper()}  ({len(out):,} rated, weights {wd}/{wt}) ===")
        cols = ["player", "country", "innings", "average", "cri_plus", "sr_plus", "bat_plus"]
        cols = [c for c in cols if c in out.columns]
        print(out.head(12)[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    if frames:
        allf = pd.concat(frames, ignore_index=True)
        allf.to_parquet(PROC / "cri_plus_all.parquet", index=False)
        print(f"\nwrote {PROC / 'cri_plus_all.parquet'} ({len(allf):,} rows)")


if __name__ == "__main__":
    main()
