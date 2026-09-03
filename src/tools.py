"""Data tools the debate agents call.

Every factual claim in a Howzat verdict has to come back to one of these
functions. An agent assertion with no tool result behind it is treated as
unsupported, which is what stops a confident multi-agent debate from
manufacturing consensus out of nothing.
"""

from __future__ import annotations

import difflib
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


# Women's formats must be listed here. Without them, format="wodi" fell through
# to the men's Test default and "Mithali" resolved to Wasim Raja -- the same
# silent wrong-format failure that once served Test ratings for ODI questions.
FORMATS = ("test", "odi", "t20i", "wtest", "wodi", "wt20i")


def _check_fmt(fmt: str | None) -> str:
    """Validate a format instead of quietly substituting the default.

    Coercing an unknown format to "test" is how format='wodi' ended up serving
    men's Test ratings and resolving 'Mithali' to Wasim Raja. An unknown format
    is a caller error and should say so.
    """
    f = (fmt or "test").lower()
    if f not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {list(FORMATS)}")
    return f


@lru_cache(maxsize=8)
def _ratings(fmt: str = "test") -> pd.DataFrame:
    """Ratings for one format.

    Serving Test ratings for an ODI question was a real bug: the debate panel
    argued about Viv Richards' ODI standing using his Test numbers, and the
    Skeptic caught it. Every lookup now takes an explicit format.
    """
    fmt = _check_fmt(fmt)
    for name in (f"cri_plus_{fmt}.parquet", "cri_plus.parquet" if fmt == "test" else ""):
        if name and (PROC / name).exists():
            return pd.read_parquet(PROC / name)
    raise FileNotFoundError(f"no ratings for format {fmt!r}")


@lru_cache(maxsize=8)
def _innings(fmt: str = "test") -> pd.DataFrame:
    """Raw innings, filtered exactly as the ratings are.

    Without the same Full Member filter, era_context reported a 2020s T20I
    average of 19.19 while the ratings were fitted against 21.83. The panel
    would then reason about an era baseline that does not correspond to the
    numbers it is comparing.
    """
    fmt = _check_fmt(fmt)
    df = pd.read_parquet(PROC / f"{fmt}_innings.parquet")
    from formats import FULL_MEMBERS

    return df[df["opposition"].isin(FULL_MEMBERS)]


@lru_cache(maxsize=8)
def _bowling(fmt: str = "test") -> pd.DataFrame | None:
    """Bowling ratings for one format, or None if not built yet."""
    fmt = _check_fmt(fmt)
    path = PROC / f"bowl_plus_{fmt}.parquet"
    return pd.read_parquet(path) if path.exists() else None


# Statsguru stores "initials surname", so first names and nicknames never
# resolve on their own. Fans overwhelmingly use them.
ALIASES = {
    "sachin": "Tendulkar", "virat": "Kohli", "viv": "IVA Richards",
    "sunny": "Gavaskar", "sunil": "Gavaskar", "jammy": "Dravid",
    "rahul dravid": "Dravid", "the wall": "Dravid", "don": "Bradman",
    "the don": "Bradman", "brian": "Lara", "ricky": "Ponting",
    "steve smith": "SPD Smith", "steven smith": "SPD Smith",
    "joe root": "JE Root", "kane": "Williamson", "sanga": "Sangakkara",
    "kumar": "Sangakkara", "mahela": "Jayawardene", "jacques": "Kallis",
    "ab de villiers": "AB de Villiers", "abd": "AB de Villiers",
    "inzamam": "Inzamam-ul-Haq", "javed": "Javed Miandad",
    "sobers": "GS Sobers", "hobbs": "JB Hobbs", "hammond": "WR Hammond",
    "gooch": "GA Gooch", "border": "AR Border", "waugh": "SR Waugh",
    "sehwag": "Sehwag", "virender": "Sehwag", "rohit": "RG Sharma",
    "cheteshwar": "Pujara", "pujara": "Pujara", "babar": "Babar Azam",
    # Women's cricket. The table had none, so "Mithali" resolved to nothing.
    "mithali": "M Raj", "mithali raj": "M Raj",
    "meg": "MM Lanning", "meg lanning": "MM Lanning", "lanning": "MM Lanning",
    "ellyse": "EA Perry", "ellyse perry": "EA Perry", "perry": "EA Perry",
    "smriti": "S Mandhana", "mandhana": "S Mandhana",
    "alyssa": "AJ Healy", "healy": "AJ Healy",
    "belinda": "BJ Clark", "belinda clark": "BJ Clark",
    "sophie devine": "SFM Devine", "devine": "SFM Devine",
    "ecclestone": "S Ecclestone", "sophie ecclestone": "S Ecclestone",
    "fitzpatrick": "CL Fitzpatrick", "cathryn": "CL Fitzpatrick",
    "harmanpreet": "H Kaur", "shafali": "Shafali Verma",
    "bakewell": "E Bakewell", "enid": "E Bakewell",
    "charlotte": "CM Edwards", "charlotte edwards": "CM Edwards",
    "suzie": "SW Bates", "bates": "SW Bates",
    "stafanie": "SR Taylor", "beth mooney": "BL Mooney", "mooney": "BL Mooney",
}


def _resolve(name: str, fmt: str = "test") -> str | None:
    """Map a loose name ('Sachin', 'kohli') to the Statsguru player key."""
    r = _ratings(fmt)
    names = r["player"].tolist()
    low = name.strip().lower()

    if len(low) < 3:
        return None

    if low in ALIASES:
        low = ALIASES[low].lower()

    exact = [n for n in names if n.lower() == low]
    if exact:
        return exact[0]

    # Surname match: the query must match a whole name token, not merely appear
    # inside one. Without this, "Test" fuzzy-matched onto a real player and the
    # debate silently argued about the wrong person.
    toks = {n: set(re.split(r"[^a-z]+", n.lower())) for n in names}
    surname = [n for n in names if low in toks[n]]
    if surname:
        sub = r[r.player.isin(surname)].sort_values("innings", ascending=False)
        return sub.iloc[0]["player"]

    subs = [n for n in names if low in n.lower()]
    if subs:
        sub = r[r.player.isin(subs)].sort_values("innings", ascending=False)
        return sub.iloc[0]["player"]

    close = difflib.get_close_matches(name, names, n=1, cutoff=0.85)
    return close[0] if close else None


def search_players(query: str, limit: int = 8, format: str = "test") -> list[dict]:
    """Fuzzy-find players by name."""
    r = _ratings(format)
    low = query.strip().lower()
    hit = r[r.player.str.lower().str.contains(low, regex=False)]
    if hit.empty:
        names = difflib.get_close_matches(query, r.player.tolist(), n=limit, cutoff=0.5)
        hit = r[r.player.isin(names)]
    hit = hit.sort_values("innings", ascending=False).head(limit)
    return hit[["player", "country", "innings", "average", "cri_plus"]].to_dict("records")


def get_player(name: str, format: str = "test") -> dict:
    """Career record and rating for one batter, in the given format."""
    try:
        fmt = _check_fmt(format)
    except ValueError as e:
        return {"error": str(e)}
    key = _resolve(name, fmt)
    if key is None:
        return {
            "error": f"no {fmt.upper()} player matching {name!r}",
            "suggestions": search_players(name, format=fmt),
        }

    r = _ratings(fmt)
    row = r[r.player == key].iloc[0]
    ii = _innings(fmt)
    inn = ii[ii.player == key]

    # groupby.apply can hand back a Series rather than a frame depending on the
    # shape, which then makes sort_values("innings") raise. agg is unambiguous.
    g = inn.groupby("opposition")
    by_opp = pd.DataFrame({
        "innings": g.size(),
        "runs": g.runs.sum().astype(int),
        "not_outs": g.not_out.sum().astype(int),
    })
    by_opp["average"] = (
        by_opp.runs / (by_opp.innings - by_opp.not_outs).clip(lower=1)
    ).round(2)
    by_opp = by_opp.drop(columns=["not_outs"]).sort_values("innings", ascending=False)

    main = float(row["bat_plus"]) if "bat_plus" in r.columns else float(row.cri_plus)
    mname = "bat_plus" if "bat_plus" in r.columns else "cri_plus"
    out = {
        "format": fmt.upper(),
        "player": key,
        "country": row.country,
        "span": f"{int(row.first_year)}-{int(row.last_year)}",
        "innings": int(row.innings),
        "runs": int(row.runs),
        "not_outs": int(row.not_outs),
        "average": round(float(row.average), 2),
        "cri_plus": round(float(row.cri_plus), 1),
        "cri_plus_95_low": round(float(row.cri_lo), 1),
        "cri_plus_95_high": round(float(row.cri_hi), 1),
        "cri_plus_rank": int((_ratings().cri_plus > row.cri_plus).sum() + 1),
        "highest_score": int(inn.runs.max()) if len(inn) else None,
        "hundreds": int((inn.runs >= 100).sum()),
        "fifties": int(((inn.runs >= 50) & (inn.runs < 100)).sum()),
        "ducks": int((inn.runs == 0).sum()),
        "by_opposition": by_opp.head(8).reset_index().to_dict("records"),
    }
    out["rating_metric"] = mname
    out["rating"] = round(main, 1)
    out["rating_rank"] = int((r[mname] > row[mname]).sum() + 1)
    if "sr_plus" in r.columns and pd.notna(row.get("sr_plus")):
        out["sr_plus"] = round(float(row.sr_plus), 1)
    return out


def compare(names: list[str], format: str = "test") -> dict:
    """Side-by-side comparison, the core call for a GOAT argument."""
    try:
        fmt = _check_fmt(format)
    except ValueError as e:
        return {"error": str(e)}
    if len(set(n.strip().lower() for n in names)) < 2:
        return {"error": "compare needs two different players"}
    players = [get_player(n, format=fmt) for n in names]
    ok = [p for p in players if "error" not in p]
    # Keep the payload compact: the per-opposition breakdown is large and is
    # available via get_player. Including it here overflowed the tool-result
    # budget and silently truncated the third player out of the comparison.
    slim = [{k: v for k, v in p.items() if k != "by_opposition"} for p in ok]
    if len(ok) < 2:
        return {"error": "need at least two resolvable players", "results": players}

    ranked = sorted(ok, key=lambda p: -p["rating"])
    lead, second = ranked[0], ranked[1]
    separated = lead["cri_plus_95_low"] > second["cri_plus_95_high"]
    return {
        "format": fmt.upper(),
        "players": slim,
        "rating_order": [p["player"] for p in ranked],
        "leader": lead["player"],
        "margin_over_next": round(lead["rating"] - second["rating"], 1),
        "intervals_separated": bool(separated),
        "separation_note": (
            "The leader's 95% interval clears the runner-up's: the gap is established."
            if separated else
            "The top two 95% intervals OVERLAP: this data cannot separate them. "
            "Do not declare a confident winner."
        ),
        "raw_average_order": [p["player"] for p in sorted(ok, key=lambda p: -p["average"])],
        "note": (
            "cri_plus_order and raw_average_order differing is the era "
            "adjustment doing work; 100 = average Test batter of the same era."
        ),
    }


def era_context(decade: int, format: str = "test") -> dict:
    """How hard was scoring in a given decade, in a given format?"""
    try:
        inn = _innings(_check_fmt(format))
    except (ValueError, FileNotFoundError) as e:
        return {"error": str(e)}
    d = inn[inn.year // 10 * 10 == decade]
    if d.empty:
        return {"error": f"no innings in {decade}s"}
    dismissals = max(len(d) - int(d.not_out.sum()), 1)
    return {
        "format": (format or "test").upper(),
        "decade": f"{decade}s",
        "innings": int(len(d)),
        "batting_average": round(float(d.runs.sum()) / float(dismissals), 2),
        "hundred_rate": round(float((d.runs >= 100).mean()), 4),
        "duck_rate": round(float((d.runs == 0).mean()), 4),
    }


def get_bowler(name: str, format: str = "test") -> dict:
    """Career bowling record and BOWL+ rating, in the given format."""
    try:
        fmt = _check_fmt(format)
    except ValueError as e:
        return {"error": str(e)}
    b = _bowling(fmt)
    if b is None:
        return {
            "error": f"{fmt.upper()} bowling ratings not built yet",
            "available": [f for f in FORMATS if _bowling(f) is not None],
        }

    names = b["player"].tolist()
    low = name.strip().lower()
    if low in ALIASES:
        low = ALIASES[low].lower()
    if len(low) < 3:
        return {"error": f"name {name!r} too short"}

    toks = {n: set(re.split(r"[^a-z]+", n.lower())) for n in names}
    hit = [n for n in names if n.lower() == low] or [n for n in names if low in toks[n]]
    if not hit:
        hit = [n for n in names if low in n.lower()]
    if not hit:
        return {"error": f"no bowler matching {name!r}"}

    sub = b[b.player.isin(hit)].sort_values("wickets", ascending=False)
    row = sub.iloc[0]
    return {
        "format": fmt.upper(),
        "player": row.player,
        "country": row.country,
        "span": f"{int(row.first_year)}-{int(row.last_year)}",
        "wickets": int(row.wickets),
        "balls": int(row.balls),
        "bowling_average": round(float(row.bowl_average), 2),
        "economy": round(float(row.economy), 2),
        "strike_rate": round(float(row.strike_rate), 1),
        "bowl_plus": round(float(row.bowl_plus), 1),
        "bowl_plus_rank": int((b.bowl_plus > row.bowl_plus).sum() + 1),
        "note": "BOWL+ 100 = an average Test bowler of the same era; higher is better.",
    }


def bowling_leaderboard(top: int = 20, format: str = "test") -> list[dict] | dict:
    try:
        fmt = _check_fmt(format)
    except ValueError as e:
        return {"error": str(e)}
    b = _bowling(fmt)
    if b is None:
        return {"error": f"{fmt.upper()} bowling ratings not built yet"}
    cols = ["player", "country", "wickets", "bowl_average", "economy", "bowl_plus"]
    return b.nlargest(top, "bowl_plus")[cols].round(2).to_dict("records")


@lru_cache(maxsize=1)
def _rule_changes():
    path = PROC / "rule_changes.parquet"
    return pd.read_parquet(path) if path.exists() else None


def rule_change_effect(year: int | None = None, format: str | None = None) -> dict:
    """Measured effect of Law / playing-condition changes on scoring.

    Every figure carries a 95% interval and a permutation p-value. Two
    estimates are given because they can disagree, and the disagreement is
    informative: a raw before/after says covering pitches in 1972 did nothing,
    while against the pre-existing trend it was worth +2.37 runs per dismissal.
    """
    d = _rule_changes()
    if d is None:
        return {"error": "rule-change analysis not built; run src/rulechange.py"}
    if format:
        d = d[d["format"] == format.lower()]
    if year:
        d = d[(d.year - int(year)).abs() <= 5]
    if d.empty:
        return {"error": "no rule changes matched", "available_years": sorted(
            _rule_changes().year.unique().tolist())}
    return {
        "changes": d.to_dict("records"),
        "note": (
            "diff/ci_low/ci_high are the raw before-vs-after change in runs per "
            "dismissal. its_effect is the departure from the pre-existing trend. "
            "Cite both when they disagree; significant requires the interval to "
            "exclude zero AND p<0.05. These are observational, so associations "
            "rather than proven causes."
        ),
    }


def leaderboard(top: int = 20, min_innings: int = 40, format: str = "test") -> list[dict] | dict:
    try:
        r = _ratings(_check_fmt(format))
    except (ValueError, FileNotFoundError) as e:
        return {"error": str(e)}
    key = "bat_plus" if "bat_plus" in r.columns else "cri_plus"
    r = r[r.innings >= min_innings].nlargest(top, key)
    cols = ["player", "country", "innings", "average", "cri_plus", "sr_plus",
            "bat_plus", "cri_lo", "cri_hi"]
    return r[[c for c in cols if c in r.columns]].round(1).to_dict("records")


# Schema advertised to the model. Kept in one place so the agent prompt and the
# dispatcher can never drift apart.
TOOLS = {
    "search_players": search_players,
    "get_player": get_player,
    "compare": compare,
    "era_context": era_context,
    "leaderboard": leaderboard,
    "get_bowler": get_bowler,
    "bowling_leaderboard": bowling_leaderboard,
    "rule_change_effect": rule_change_effect,
}

_STR = {"type": "string"}
_INT = {"type": "integer"}

TOOL_SCHEMAS = [
    {
        "name": "get_player",
        "description": (
            "Career record and era-adjusted rating for one batter in a given "
            "FORMAT (test/odi/t20i). 100 = an average batter of the same era and "
            "format. Always pass the format the question is about."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": dict(_STR, description="Player name, e.g. 'Tendulkar'"),
                "format": dict(_STR, description="Format: test, odi or t20i. Default test."),
            },
            "required": ["name"],
        },
    },
    {
        "name": "compare",
        "description": (
            "Compare two or more batters side by side. Returns both the CRI+ order "
            "and the raw batting-average order; where they differ, the era "
            "adjustment is doing the work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": _STR,
                    "description": "Two or more player names",
                },
                "format": dict(_STR, description="Format: test, odi or t20i. Default test."),
            },
            "required": ["names"],
        },
    },
    {
        "name": "era_context",
        "description": "Scoring conditions in a decade: batting average, hundred rate, duck rate.",
        "parameters": {
            "type": "object",
            "properties": {
                "decade": dict(_INT, description="Decade start year, e.g. 1930"),
                "format": dict(_STR, description="Format: test, odi or t20i. Default test."),
            },
            "required": ["decade"],
        },
    },
    {
        "name": "get_bowler",
        "description": (
            "Career Test bowling record and BOWL+ rating for one bowler. BOWL+ is "
            "era-adjusted: 100 = an average Test bowler of the same era, higher is "
            "better. Use this for bowlers; get_player is for batters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": dict(_STR, description="Bowler name, e.g. 'Marshall'"),
                "format": dict(_STR, description="Format: test, odi or t20i. Default test."),
            },
            "required": ["name"],
        },
    },
    {
        "name": "bowling_leaderboard",
        "description": "Top Test bowlers of all time by BOWL+.",
        "parameters": {
            "type": "object",
            "properties": {
                "top": dict(_INT, description="How many, default 20"),
                "format": dict(_STR, description="Format: test, odi or t20i. Default test."),
            },
        },
    },
    {
        "name": "rule_change_effect",
        "description": (
            "Measured effect of cricket Law and playing-condition changes on "
            "scoring, with 95% confidence intervals and permutation p-values. "
            "Use this whenever a question involves era comparison, whether a "
            "rule made batting or bowling easier, or why scoring shifted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "year": dict(_INT, description="Focus on changes near this year"),
                "format": dict(_STR, description="test, odi or t20i"),
            },
        },
    },
    {
        "name": "search_players",
        "description": "Fuzzy-search players by name when a name does not resolve.",
        "parameters": {
            "type": "object",
            "properties": {"query": _STR},
            "required": ["query"],
        },
    },
    {
        "name": "leaderboard",
        "description": "Top batters of all time by CRI+.",
        "parameters": {
            "type": "object",
            "properties": {
                "top": dict(_INT, description="How many to return, default 20"),
                "min_innings": dict(_INT, description="Minimum innings, default 40"),
                "format": dict(_STR, description="Format: test, odi or t20i. Default test."),
            },
        },
    },
]


def dispatch(name: str, args: dict) -> dict:
    """Run a tool call from an agent, never raising into the debate loop."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}", "available": list(TOOLS)}
    try:
        return {"result": fn(**args)}
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:  # a broken tool must not kill the debate
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    import json

    print(json.dumps(get_player("Bradman"), indent=2, default=str)[:900])
    print(json.dumps(compare(["Bradman", "Tendulkar", "Kohli"]), indent=2, default=str)[:900])
