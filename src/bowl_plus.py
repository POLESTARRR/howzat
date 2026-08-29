"""BOWL+ : an era-adjusted bowling rating.

Bowling average (runs conceded per wicket) has the same era problem as batting
average, in the opposite direction: an average of 25 in 2005 is not an average
of 25 in 1955. It also hides *how* a bowler is good. Two bowlers with identical
averages can differ completely in whether they buy wickets cheaply or bowl long
economical spells.

So BOWL+ decomposes into the two things a bowler actually does, each modelled
per ball and each era-adjusted:

    wickets_i ~ Poisson(balls_i * exp(strike[bowler] + era + opp))
    runs_i    ~ Poisson(balls_i * exp(econ[bowler]   + era + opp))

Bowling average is exactly the ratio of these rates, so a bowler's overall skill
is `strike - econ` in log space: taking wickets faster and conceding fewer runs
both help, and the two trade off exactly as they do in the real statistic.

    BOWL+ = 100 * exp(strike - econ - reference)

Higher is better, and 100 is an average bowler of the same era, matching CRI+.
Era terms are fixed offsets from observed scoring for the same identification
reason as in `cri_plus` -- fitted freely they are confounded with bowler skill.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

MIN_BALLS = 3000  # roughly 500 overs: enough for a rating to mean something


def load_bowling(fmt: str = "test") -> pd.DataFrame:
    path = PROC / f"{fmt}_bowling.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run: python3 src/fetch_statsguru.py {fmt} "
            f"1877 2026 bowling"
        )
    df = pd.read_parquet(path)
    df = df.dropna(subset=["balls", "runs_conceded", "wickets", "year"]).copy()
    df = df[df["balls"] > 0]
    df["decade"] = (df["year"] // 10 * 10).astype(int)
    return df


def opp_era_offsets(df: pd.DataFrame) -> pd.Series:
    """How strongly each side batted, per decade, relative to that decade.

    A single opposition constant across 150 years is wrong, and expensively so.
    South Africa fits at -0.139 ("hard to take wickets against") because MODERN
    South Africa is strong -- which then penalised George Lohmann for bowling at
    1890s South Africa, the weakest batting side in Test history, against whom
    he averaged 5.80. Opposition strength is strongly time-varying.

    Measured relative to the decade, so it does not double-count the era term.
    """
    g = df.groupby(["opposition", "decade"])
    team = g["runs_conceded"].sum() / g["wickets"].sum().clip(lower=1)
    d = df.groupby("decade")
    era = d["runs_conceded"].sum() / d["wickets"].sum().clip(lower=1)
    # Enough evidence, or fall back to neutral.
    n = g["wickets"].sum()
    off = np.log(team / team.index.get_level_values("decade").map(era))
    return off.where(n >= 40, 0.0)


def era_rates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Observed log wicket-rate and log run-rate per ball, per decade."""
    g = df.groupby("decade")
    balls = g["balls"].sum().clip(lower=1)
    w = np.log((g["wickets"].sum().clip(lower=1)) / balls)
    r = np.log((g["runs_conceded"].sum().clip(lower=1)) / balls)
    return w - w.mean(), r - r.mean()


class BowlPlusModel:
    """Two coupled Poisson rate models, fitted jointly with analytic gradients."""

    def __init__(self, ridge: float = 1.0, ridge_ctx: float = 0.05):
        self.ridge = ridge
        self.ridge_ctx = ridge_ctx

    def _design(self, df: pd.DataFrame):
        self.bowlers = pd.Index(sorted(df["player"].unique()))
        self.decades = pd.Index(sorted(df["decade"].unique()))
        self.opps = pd.Index(sorted(df["opposition"].unique()))
        return (
            self.bowlers.get_indexer(df["player"]),
            self.decades.get_indexer(df["decade"]),
            self.opps.get_indexer(df["opposition"]),
        )

    def _opp_offset(self, df: pd.DataFrame) -> np.ndarray:
        """Per-innings offset for how well that side batted, in that decade."""
        off = opp_era_offsets(df)
        idx = pd.MultiIndex.from_arrays([df["opposition"], df["decade"]])
        return off.reindex(idx).fillna(0.0).to_numpy()

    def _obj(self, theta, bi, di, oi, balls, wkts, runs, nb, nd, no):
        """Negative penalised log-likelihood of both Poisson models, plus gradient."""
        s = theta[:nb]                      # wicket-taking skill
        e = theta[nb : 2 * nb]              # economy skill (higher = leakier)
        i_w, i_r = theta[2 * nb], theta[2 * nb + 1]
        ow_raw = theta[2 * nb + 2 : 2 * nb + 2 + no]
        or_raw = theta[2 * nb + 2 + no :]
        ow = ow_raw - ow_raw.mean()
        orr = or_raw - or_raw.mean()

        logb = np.log(balls)
        # A strong batting side both survives longer and scores faster, so the
        # same offset enters the wicket model negatively and the run model
        # positively.
        eta_w = np.clip(i_w + s[bi] + self.era_w_[di] + ow[oi] - self.oppoff_, -12, 4)
        eta_r = np.clip(i_r + e[bi] + self.era_r_[di] + orr[oi] + self.oppoff_, -12, 4)
        lam_w = np.exp(eta_w + logb)
        lam_r = np.exp(eta_r + logb)

        # Poisson log-lik, dropping the constant log(k!) term.
        ll = (wkts * eta_w - lam_w) + (runs * eta_r - lam_r)

        pen = (
            self.ridge * (s @ s)
            + self.ridge * (e @ e)
            + self.ridge_ctx * (ow @ ow + orr @ orr)
        )
        f = -ll.sum() + pen

        gw = wkts - lam_w          # d ll / d eta_w
        gr = runs - lam_r

        g = np.empty_like(theta)
        g[:nb] = -np.bincount(bi, gw, minlength=nb) + 2 * self.ridge * s
        g[nb : 2 * nb] = -np.bincount(bi, gr, minlength=nb) + 2 * self.ridge * e
        g[2 * nb] = -gw.sum()
        g[2 * nb + 1] = -gr.sum()

        gow = -np.bincount(oi, gw, minlength=no) + 2 * self.ridge_ctx * ow
        gor = -np.bincount(oi, gr, minlength=no) + 2 * self.ridge_ctx * orr
        g[2 * nb + 2 : 2 * nb + 2 + no] = gow - gow.mean()
        g[2 * nb + 2 + no :] = gor - gor.mean()
        return f, g

    def fit(self, df: pd.DataFrame) -> "BowlPlusModel":
        bi, di, oi = self._design(df)
        ew, er = era_rates(df)
        self.era_w_ = ew.reindex(self.decades).fillna(0.0).to_numpy()
        self.era_r_ = er.reindex(self.decades).fillna(0.0).to_numpy()
        self.oppoff_ = self._opp_offset(df)

        balls = df["balls"].to_numpy(float)
        wkts = df["wickets"].to_numpy(float)
        runs = df["runs_conceded"].to_numpy(float)

        nb, nd, no = len(self.bowlers), len(self.decades), len(self.opps)
        th = np.zeros(2 * nb + 2 + 2 * no)
        th[2 * nb] = np.log(max(wkts.sum() / balls.sum(), 1e-6))
        th[2 * nb + 1] = np.log(max(runs.sum() / balls.sum(), 1e-6))

        res = minimize(
            self._obj, th,
            args=(bi, di, oi, balls, wkts, runs, nb, nd, no),
            method="L-BFGS-B", jac=True,
            options={"maxiter": 2000, "maxfun": 20000},
        )
        self.result_ = res
        t = res.x
        self.strike_ = pd.Series(t[:nb], index=self.bowlers)
        self.econ_ = pd.Series(t[nb : 2 * nb], index=self.bowlers)
        self.i_w_, self.i_r_ = t[2 * nb], t[2 * nb + 1]
        ow = t[2 * nb + 2 : 2 * nb + 2 + no]
        orr = t[2 * nb + 2 + no :]
        self.opp_w_ = pd.Series(ow - ow.mean(), index=self.opps)
        self.opp_r_ = pd.Series(orr - orr.mean(), index=self.opps)
        return self

    def fit_eb(self, df: pd.DataFrame, iters: int = 10, tol: float = 1e-3):
        for _ in range(iters):
            self.fit(df)
            var = max(
                float(np.mean(self.strike_.to_numpy() ** 2 + self.econ_.to_numpy() ** 2) / 2),
                1e-4,
            )
            new = 1.0 / (2.0 * var)
            done = abs(new - self.ridge) / max(new, 1e-9) < tol
            self.ridge = new
            if done:
                break
        return self.fit(df)

    def skill_se(self, df: pd.DataFrame) -> pd.Series:
        """Standard error of (strike - econ) from the observed information.

        For a Poisson with a log link the second derivative w.r.t. eta is just
        lambda, so the per-bowler information is the sum of fitted counts plus
        the prior curvature. The two rate models contribute independently, so
        their variances add. This is what separates an 85-wicket career from a
        563-wicket one: same point estimate, very different confidence.
        """
        bi, di, oi = self._design(df)
        balls = df["balls"].to_numpy(float)
        nb = len(self.bowlers)
        ow = self.opp_w_.reindex(self.opps).fillna(0.0).to_numpy()
        orr = self.opp_r_.reindex(self.opps).fillna(0.0).to_numpy()
        oppoff = self._opp_offset(df)

        logb = np.log(balls)
        lam_w = np.exp(np.clip(self.i_w_ + self.strike_.to_numpy()[bi]
                               + self.era_w_[di] + ow[oi] - oppoff, -12, 4) + logb)
        lam_r = np.exp(np.clip(self.i_r_ + self.econ_.to_numpy()[bi]
                               + self.era_r_[di] + orr[oi] + oppoff, -12, 4) + logb)

        h_w = np.bincount(bi, lam_w, minlength=nb) + 2 * self.ridge
        h_r = np.bincount(bi, lam_r, minlength=nb) + 2 * self.ridge
        var = 1.0 / h_w + 1.0 / h_r
        return pd.Series(np.sqrt(var), index=self.bowlers)

    def ratings(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        skill = self.strike_ - self.econ_
        skill = skill - skill.mean()
        out = pd.DataFrame(
            {"strike_skill": self.strike_, "econ_skill": self.econ_,
             "bowl_plus": 100 * np.exp(skill)}
        )
        if df is not None:
            se = self.skill_se(df)
            out["bowl_lo"] = 100 * np.exp(skill - 1.96 * se)
            out["bowl_hi"] = 100 * np.exp(skill + 1.96 * se)
        return out.rename_axis("player").reset_index()


def careers(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("player")
    out = g.agg(
        country=("country", lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0]),
        innings=("balls", "size"),
        balls=("balls", "sum"),
        runs_conceded=("runs_conceded", "sum"),
        wickets=("wickets", "sum"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    )
    out["bowl_average"] = out.runs_conceded / out.wickets.clip(lower=1)
    out["economy"] = out.runs_conceded / (out.balls / 6).clip(lower=1)
    out["strike_rate"] = out.balls / out.wickets.clip(lower=1)
    return out.reset_index()


def build(fmt: str = "test", min_balls: int = MIN_BALLS):
    df = load_bowling(fmt)
    c = careers(df)
    keep = c.loc[c["balls"] >= min_balls, "player"]
    sub = df[df["player"].isin(keep)].copy()

    model = BowlPlusModel().fit_eb(sub)
    out = c.merge(model.ratings(sub), on="player", how="inner")
    out = out.sort_values("bowl_plus", ascending=False).reset_index(drop=True)
    out.to_parquet(PROC / f"bowl_plus_{fmt}.parquet", index=False)
    return df, model, out


if __name__ == "__main__":
    import sys

    fmt = sys.argv[1] if len(sys.argv) > 1 else "test"
    df, model, out = build(fmt)
    print(f"{len(df):,} bowling innings -> {len(out):,} rated (>= {MIN_BALLS} balls)")
    print(f"converged={model.result_.success}  EB ridge={model.ridge:.3f}\n")
    cols = ["player", "country", "wickets", "bowl_average", "bowl_plus",
            "bowl_lo", "bowl_hi"]
    print(out.head(18)[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\nInterval width vs sample size:")
    for lo, hi, lbl in [(0, 120, "<120 wkts"), (120, 300, "120-299"), (300, 9999, "300+")]:
        sel = out[(out.wickets >= lo) & (out.wickets < hi)]
        if len(sel):
            print(f"  {lbl:10s} n={len(sel):3d}  mean 95% CI width = "
                  f"{(sel.bowl_hi - sel.bowl_lo).mean():6.1f}")
