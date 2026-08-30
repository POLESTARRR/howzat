"""Peak ratings: a batter's best sustained window, not their career average.

Career CRI+ answers "how good were they overall". It is the wrong tool for
"how good were they at their best", and GOAT arguments are usually about the
latter. A long tail of decline drags a career number down; a short career
cannot show a peak at all.

Method
------
Refitting the whole model per window would be wasteful and would let each
window redefine its own era baseline. Instead the fitted era and opposition
effects are held fixed as offsets, and only the player's skill is re-estimated
inside the window. That is a one-parameter MLE per window:

    maximise  sum_i  loglik(runs_i | eta = c + era_i + opp_i + intercept)

over scalar c, with the same Gaussian prior (ridge) as the full model so short
windows are still shrunk. Solved by Newton on the score function, which is
monotone here, so it converges in a handful of steps.

Peak = the best-scoring window meeting a minimum-innings bar.

A caveat worth stating plainly. Shrinkage is evidence-proportional, so a window
holding fewer innings is pulled harder toward the mean than a whole career. For
a short career whose best window *is* most of the career, that can put the peak
estimate slightly BELOW the career estimate. This is correct Bayesian behaviour,
not an arithmetic slip, but a "peak" that reads lower than the career is
misleading, so those cases are flagged `peak_distinct = False` rather than
quietly published or fudged upward.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cri_plus import CriPlusModel, career_table

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# A peak window must be long enough to be sustained but short enough to be a
# peak. Limited-overs careers are shorter and denser, so the innings bar scales
# with the format rather than being one Test-sized number.
WINDOW_YEARS = 4
MIN_WINDOW_INNINGS_BY_FORMAT = {
    "test": 20, "odi": 25, "t20i": 20,
    "wtest": 6, "wodi": 20, "wt20i": 15,
}
MIN_WINDOW_INNINGS = MIN_WINDOW_INNINGS_BY_FORMAT["test"]
MIN_CAREER_INNINGS = {
    "test": 20, "odi": 25, "t20i": 20,
    "wtest": 8, "wodi": 25, "wt20i": 20,
}


def _skill_mle(runs, censored, offset, ridge, iters: int = 60) -> float:
    """One-parameter censored-geometric MLE for skill, given fixed offsets."""
    c = 0.0
    for _ in range(iters):
        eta = np.clip(c + offset, -4.0, 6.0)
        mu = np.exp(eta)
        inv = 1.0 / (1.0 + mu)
        # score and its derivative w.r.t. c
        g = np.where(censored, (runs + 1) * inv, (runs - mu) * inv).sum() - 2 * ridge * c
        h = -(mu * (1.0 + runs) * inv * inv).sum() - 2 * ridge
        if not np.isfinite(h) or abs(h) < 1e-12:
            break
        step = g / h
        c -= step
        if abs(step) < 1e-9:
            break
    return float(c)


def peak_ratings(
    fmt: str = "test",
    window: int = WINDOW_YEARS,
    min_window_innings: int | None = None,
) -> pd.DataFrame:
    from formats import load_format

    if min_window_innings is None:
        min_window_innings = MIN_WINDOW_INNINGS_BY_FORMAT.get(fmt, MIN_WINDOW_INNINGS)
    df = load_format(fmt)
    careers = career_table(df)
    keep = careers.loc[careers["innings"] >= MIN_CAREER_INNINGS.get(fmt, 20), "player"]
    sub = df[df["player"].isin(keep)].copy()

    model = CriPlusModel().fit_eb(sub)

    era = model.era_.reindex(model.decades)
    opp = model.opp_.reindex(model.opps)
    sub = sub.assign(
        offset=model.intercept_
        + sub["decade"].map(era).fillna(0.0).to_numpy()
        + sub["opposition"].map(opp).fillna(0.0).to_numpy()
    )

    rows = []
    for player, g in sub.groupby("player", sort=False):
        yrs = g["year"].to_numpy()
        best = None
        for start in range(int(yrs.min()), int(yrs.max()) - window + 2):
            m = (yrs >= start) & (yrs < start + window)
            n = int(m.sum())
            if n < min_window_innings:
                continue
            c = _skill_mle(
                g["runs"].to_numpy(float)[m],
                g["not_out"].to_numpy(bool)[m],
                g["offset"].to_numpy()[m],
                model.ridge_player,
            )
            if best is None or c > best[0]:
                best = (c, start, n)
        if best is None:
            continue
        c, start, n = best
        rows.append({
            "player": player,
            "peak_plus": 100 * np.exp(c),
            "peak_start": start,
            "peak_end": start + window - 1,
            "peak_innings": n,
        })

    peak = pd.DataFrame(rows)
    out = careers.merge(peak, on="player", how="inner")
    out = out.merge(model.ratings()[["player", "cri_plus"]], on="player", how="left")
    out["peak_lift"] = out["peak_plus"] - out["cri_plus"]
    # A peak is only meaningful if the window is a real subset of the career.
    # Otherwise the "peak" is just the career measured with less evidence.
    out["peak_share"] = out["peak_innings"] / out["innings"]
    out["peak_distinct"] = (out["peak_share"] <= 0.80) & (out["peak_lift"] >= 0)
    out["format"] = fmt
    out = out.sort_values("peak_plus", ascending=False).reset_index(drop=True)
    out.to_parquet(PROC / f"peak_{fmt}.parquet", index=False)
    return out


if __name__ == "__main__":
    import sys

    fmt = sys.argv[1] if len(sys.argv) > 1 else "test"
    out = peak_ratings(fmt)
    cols = ["player", "country", "innings", "cri_plus", "peak_plus",
            "peak_start", "peak_end", "peak_innings"]
    print(f"{fmt.upper()}: {len(out):,} players with a qualifying "
          f"{WINDOW_YEARS}-year window\n")
    print("TOP 15 BY PEAK")
    print(out.head(15)[cols].to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    print("\nBIGGEST PEAK-OVER-CAREER LIFT (min 80 inns)")
    q = out[out.innings >= 80]
    print(q.nlargest(10, "peak_lift")[cols + ["peak_lift"]].to_string(
        index=False, float_format=lambda v: f"{v:.0f}"))
