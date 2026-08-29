"""CRI+ : an era-adjusted Test batting rating.

Raw batting average is not comparable across eras. Two things break it:

1. Era difficulty drifts (uncovered pitches, bat technology, ground size,
   bowling standards), so 50 in 1955 is not 50 in 2005.
2. Not-outs are treated as if the innings simply ended, which inflates the
   average. They are really *right-censored* observations: we know the batter
   scored at least R, not exactly R.

Model
-----
Runs in a Test innings are close to geometrically distributed: roughly constant
hazard of dismissal per ball. We therefore model innings runs as Geometric with
mean mu, and let mu depend on who is batting, when, and against whom:

    log mu_i = intercept + skill[player_i] + era[decade_i] + opp[opposition_i]

Dismissed innings contribute the pmf; not-outs contribute the survival function
P(X > r). Player skills carry a Gaussian prior (ridge penalty), which is partial
pooling: a batter with 8 innings is shrunk toward the mean, a batter with 200 is
not. That shrinkage is what stops small-sample flukes outranking Tendulkar.

CRI+ is then indexed so 100 = the era-average batter:

    CRI+ = 100 * exp(skill_player)

Interpretation matches wRC+ in baseball: 130 means 30% better than an average
Test batter *of the same era*, and the number is comparable across all eras.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# A batter needs enough innings for a rating to mean anything.
MIN_INNINGS = 20


def canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    """Use Statsguru's player id as identity, with one display name per id.

    Names are neither unique nor stable. Pakistan has fielded two "Imran
    Khan"s, which merged into a single 1971-2019 career with 391 wickets
    instead of the great one's 362. In the other direction, Richard Hadlee
    appears as both "RJ Hadlee" and "Sir RJ Hadlee" after his knighthood, which
    split one career in two. Keying on the id fixes both.
    """
    if "player_id" not in df.columns or df["player_id"].isna().all():
        return df
    df = df.dropna(subset=["player_id"]).copy()
    # Prefer the longest name seen for an id: "Sir RJ Hadlee" over "RJ Hadlee".
    name = (df.groupby("player_id")["player"]
              .agg(lambda s: max(s.unique(), key=len)))

    # Two ids can still share a name. Disambiguate with the debut year, or the
    # two Imran Khans re-merge the moment anything groups by name.
    debut = df.groupby("player_id")["year"].min()
    dupes = name[name.duplicated(keep=False)]
    for pid in dupes.index:
        name.loc[pid] = f"{name.loc[pid]} ({int(debut.loc[pid])})"

    df["player"] = df["player_id"].map(name)
    return df


def load_innings() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "test_innings.parquet")
    df = df.dropna(subset=["runs", "year"]).copy()
    df["decade"] = (df["year"] // 10 * 10).astype(int)
    return canonicalise(df)


# ---------------------------------------------------------------- descriptive


def era_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Era-average batting average, the naive baseline CRI+ must beat."""
    g = df.groupby("decade")
    out = g.agg(
        innings=("runs", "size"),
        runs=("runs", "sum"),
        not_outs=("not_out", "sum"),
    )
    out["dismissals"] = out["innings"] - out["not_outs"]
    out["era_average"] = out["runs"] / out["dismissals"]
    return out.reset_index()


def career_table(df: pd.DataFrame) -> pd.DataFrame:
    key = "player_id" if "player_id" in df.columns else "player"
    g = df.groupby(key)
    out = g.agg(
        # Most-frequent country, not first: players appearing for composite sides
        # (Africa XI, World XI, ICC XI) were otherwise labelled by those.
        country=("country", lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0]),
        player_name=("player", lambda s: max(s.unique(), key=len)),
        innings=("runs", "size"),
        runs=("runs", "sum"),
        not_outs=("not_out", "sum"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    )
    out["dismissals"] = (out["innings"] - out["not_outs"]).clip(lower=1)
    out["average"] = out["runs"] / out["dismissals"]
    out = out.reset_index()
    if key == "player_id":
        names = df.groupby("player_id")["player"].agg(lambda s: max(s.unique(), key=len))
        out["player"] = out["player_id"].map(names)
    return out


# ------------------------------------------------------------------ the model


def era_offsets(df: pd.DataFrame) -> pd.Series:
    """Observed log scoring level per decade, relative to the all-time mean.

    This is the crux of the design. If the era term is left as a free
    parameter it is confounded with player skill: the model can explain the
    1880s averaging 17.5 runs either as "a brutally hard era" or as "weak
    batters", and it has no way to tell them apart. Fitting it that way
    produced era effects *anti-correlated* with observed scoring.

    So the era term is fixed from data, exactly as wRC+ uses observed league
    average rather than estimating it. The honest consequence is that CRI+
    measures dominance over contemporaries, not absolute skill across eras --
    the same claim baseball's `+` metrics make, and the strongest claim the
    data can actually support.
    """
    g = df.groupby("decade")
    runs = g["runs"].sum()
    dismissals = (g.size() - g["not_out"].sum()).clip(lower=1)
    era_avg = runs / dismissals
    grand = df["runs"].sum() / max(len(df) - int(df["not_out"].sum()), 1)
    return np.log(era_avg / grand)


class CriPlusModel:
    """Censored geometric regression with partial pooling on player skill.

    era_mode:
      "offset" -- era level fixed from observed data (default, identified)
      "free"   -- era estimated jointly (kept for comparison; confounded)
    """

    def __init__(
        self,
        ridge_player: float = 0.35,
        ridge_ctx: float = 0.05,
        era_mode: str = "offset",
    ):
        # ridge_player is the prior SD control on skills: higher => more shrinkage.
        self.ridge_player = ridge_player
        self.ridge_ctx = ridge_ctx
        self.era_mode = era_mode

    def _design(self, df: pd.DataFrame):
        self.players = pd.Index(sorted(df["player"].unique()))
        self.decades = pd.Index(sorted(df["decade"].unique()))
        self.opps = pd.Index(sorted(df["opposition"].unique()))

        pi = self.players.get_indexer(df["player"])
        di = self.decades.get_indexer(df["decade"])
        oi = self.opps.get_indexer(df["opposition"])
        return pi, di, oi

    def _obj(self, theta, pi, di, oi, runs, censored, np_, nd_, no_):
        """Negative penalised log-likelihood and its analytic gradient.

        The gradient is essential: with ~2000 player parameters a numerical
        gradient would need 2000 extra likelihood evaluations per step.
        """
        intercept = theta[0]
        skill = theta[1 : 1 + np_]
        era_raw = theta[1 + np_ : 1 + np_ + nd_]
        opp_raw = theta[1 + np_ + nd_ :]

        # Identifiability: opposition effects are centred, so the intercept
        # carries the overall level and skill is measured against it.
        opp = opp_raw - opp_raw.mean()
        if self.fixed_era_ is None:
            era = era_raw - era_raw.mean()
        else:
            era = self.fixed_era_  # data, not a parameter

        eta = np.clip(intercept + skill[pi] + era[di] + opp[oi], -4.0, 6.0)
        mu = np.exp(eta)

        # Geometric on {0,1,2,...} with mean mu, so p = 1/(1+mu):
        #   log P(X = r) = r*log(mu/(1+mu)) + log(1/(1+mu))
        #   log P(X > r) = (r+1)*log(mu/(1+mu))     <- not-outs are censored
        log1p_mu = np.log1p(mu)
        log_q = eta - log1p_mu
        ll = np.where(censored, (runs + 1) * log_q, runs * log_q - log1p_mu)

        pen = self.ridge_player * skill @ skill + self.ridge_ctx * (opp @ opp)
        if self.fixed_era_ is None:
            pen = pen + self.ridge_ctx * (era @ era)
        f = -ll.sum() + pen

        # d(loglik)/d eta
        inv = 1.0 / (1.0 + mu)
        w = np.where(censored, (runs + 1) * inv, (runs - mu) * inv)

        g = np.empty_like(theta)
        g[0] = -w.sum()
        g[1 : 1 + np_] = -np.bincount(pi, w, minlength=np_) + 2 * self.ridge_player * skill

        g_opp = -np.bincount(oi, w, minlength=no_) + 2 * self.ridge_ctx * opp
        if self.fixed_era_ is None:
            g_era = -np.bincount(di, w, minlength=nd_) + 2 * self.ridge_ctx * era
            # chain rule through the centring: d(x - mean x)/dx_j = I - 1/n
            g[1 + np_ : 1 + np_ + nd_] = g_era - g_era.mean()
        else:
            g[1 + np_ : 1 + np_ + nd_] = 0.0  # era is fixed, no gradient
        g[1 + np_ + nd_ :] = g_opp - g_opp.mean()

        return f, g

    def fit(self, df: pd.DataFrame) -> "CriPlusModel":
        pi, di, oi = self._design(df)
        runs = df["runs"].to_numpy(float)
        censored = df["not_out"].to_numpy(bool)

        if self.era_mode == "offset":
            off = era_offsets(df)
            self.fixed_era_ = off.reindex(self.decades).fillna(0.0).to_numpy()
        else:
            self.fixed_era_ = None

        np_, nd_, no_ = len(self.players), len(self.decades), len(self.opps)
        theta0 = np.zeros(1 + np_ + nd_ + no_)
        theta0[0] = np.log(max(df["runs"].mean(), 1.0))

        res = minimize(
            self._obj,
            theta0,
            args=(pi, di, oi, runs, censored, np_, nd_, no_),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2000, "maxfun": 20000},
        )
        self.result_ = res

        t = res.x
        self.intercept_ = t[0]
        self.skill_ = pd.Series(t[1 : 1 + np_], index=self.players)
        opp = t[1 + np_ + nd_ :]
        if self.fixed_era_ is None:
            era = t[1 + np_ : 1 + np_ + nd_]
            self.era_ = pd.Series(era - era.mean(), index=self.decades)
        else:
            self.era_ = pd.Series(self.fixed_era_, index=self.decades)
        self.opp_ = pd.Series(opp - opp.mean(), index=self.opps)
        return self

    def fit_eb(self, df: pd.DataFrame, iters: int = 12, tol: float = 1e-3) -> "CriPlusModel":
        """Fit with the shrinkage strength estimated from the data.

        A ridge penalty lam * s^2 is equivalent to a Gaussian prior
        N(0, sigma^2) on player skill with lam = 1 / (2 sigma^2). Rather than
        hand-picking lam, iterate: fit, re-estimate the spread of true skill,
        refit. Hand-picked shrinkage was letting 31-innings careers into the
        all-time top ten, which is a sample-size artefact, not a finding.
        """
        for _ in range(iters):
            self.fit(df)
            var = float(np.mean(self.skill_.to_numpy() ** 2))
            var = max(var, 1e-4)
            new_lam = 1.0 / (2.0 * var)
            if abs(new_lam - self.ridge_player) / max(new_lam, 1e-9) < tol:
                self.ridge_player = new_lam
                break
            self.ridge_player = new_lam
        return self.fit(df)

    def skill_se(self, df: pd.DataFrame) -> pd.Series:
        """Standard errors for player skill from the observed information.

        For both dismissed and censored innings the second derivative of the
        log-likelihood w.r.t. eta works out to the same expression:

            d2(-ll)/d eta2 = mu (1 + r) / (1 + mu)^2

        Summing that per player and adding the prior curvature 2*lambda gives
        the diagonal of the penalised Hessian. Off-diagonal terms are ignored:
        the design is near block-diagonal in players, so this is a good
        approximation and it is what makes a 31-innings rating visibly less
        certain than a 280-innings one.
        """
        pi, di, oi = self._design(df)
        runs = df["runs"].to_numpy(float)
        era = self.era_.reindex(self.decades).to_numpy()
        opp = self.opp_.reindex(self.opps).to_numpy()

        eta = np.clip(
            self.intercept_ + self.skill_.to_numpy()[pi] + era[di] + opp[oi], -4.0, 6.0
        )
        mu = np.exp(eta)
        info = mu * (1.0 + runs) / (1.0 + mu) ** 2

        h = np.bincount(pi, info, minlength=len(self.players)) + 2 * self.ridge_player
        return pd.Series(1.0 / np.sqrt(h), index=self.players)

    def ratings(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        out = pd.DataFrame(
            {"skill": self.skill_, "cri_plus": 100 * np.exp(self.skill_)}
        )
        if df is not None:
            se = self.skill_se(df)
            out["cri_lo"] = 100 * np.exp(self.skill_ - 1.96 * se)
            out["cri_hi"] = 100 * np.exp(self.skill_ + 1.96 * se)
        return out.rename_axis("player").reset_index()


def build(min_innings: int = MIN_INNINGS):
    df = load_innings()
    careers = career_table(df)
    keep = careers.loc[careers["innings"] >= min_innings, "player"]
    sub = df[df["player"].isin(keep)].copy()

    model = CriPlusModel().fit_eb(sub)
    out = careers.merge(model.ratings(sub), on="player", how="inner")
    out = out.sort_values("cri_plus", ascending=False).reset_index(drop=True)
    return df, model, out


if __name__ == "__main__":
    df, model, out = build()
    print(f"innings {len(df):,}  players rated {len(out):,}")
    print(f"converged={model.result_.success}  nll={model.result_.fun:,.0f}\n")

    print("era effects (log scale, + = easier scoring):")
    print(model.era_.round(3).to_string(), "\n")

    cols = ["player", "country", "innings", "average", "cri_plus", "first_year", "last_year"]
    print("TOP 25 BY CRI+")
    print(out.head(25)[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
