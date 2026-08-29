"""Test suite for Howzat.

Run: PYTHONPATH=src python3 -m unittest discover -s tests -v

These cover the things that actually broke during development: the gradient,
the era-confounding trap, name resolution matching junk onto real players, and
the debate loop's grounding guard.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import tools as T  # noqa: E402
from cri_plus import CriPlusModel, era_offsets, load_innings  # noqa: E402
from debate import Debate, _parse_json  # noqa: E402
from mock_provider import MockProvider  # noqa: E402


class TestGradient(unittest.TestCase):
    """The analytic gradient is load-bearing; a wrong one fails silently."""

    def _check(self, era_mode: str) -> float:
        rng = np.random.default_rng(7)
        n, npl, nd, no = 400, 10, 4, 5
        pi = rng.integers(0, npl, n)
        di = rng.integers(0, nd, n)
        oi = rng.integers(0, no, n)
        runs = rng.geometric(0.03, n).astype(float)
        cens = rng.random(n) < 0.15

        m = CriPlusModel(era_mode=era_mode)
        m.fixed_era_ = rng.normal(0, 0.2, nd) if era_mode == "offset" else None
        th = rng.normal(0, 0.3, 1 + npl + nd + no)
        th[0] = np.log(30)

        _, g = m._obj(th, pi, di, oi, runs, cens, npl, nd, no)
        num = np.zeros_like(th)
        eps = 1e-6
        for k in range(len(th)):
            a, b = th.copy(), th.copy()
            a[k] += eps
            b[k] -= eps
            num[k] = (
                m._obj(a, pi, di, oi, runs, cens, npl, nd, no)[0]
                - m._obj(b, pi, di, oi, runs, cens, npl, nd, no)[0]
            ) / (2 * eps)
        return float(np.max(np.abs(g - num) / (1 + np.abs(num))))

    def test_gradient_offset_mode(self):
        self.assertLess(self._check("offset"), 1e-5)

    def test_gradient_free_mode(self):
        self.assertLess(self._check("free"), 1e-5)


class TestCensoring(unittest.TestCase):
    def test_not_outs_raise_the_estimate(self):
        """A not-out is 'at least R', so it must not drag the mean down."""
        rng = np.random.default_rng(0)
        n = 300
        pi = np.zeros(n, int)
        di = np.zeros(n, int)
        oi = np.zeros(n, int)
        runs = np.full(n, 40.0)

        m = CriPlusModel(era_mode="free", ridge_player=1e-6, ridge_ctx=1e-6)
        m.fixed_era_ = None
        th = np.array([np.log(40.0), 0.0, 0.0, 0.0])

        none_cens = np.zeros(n, bool)
        all_cens = np.ones(n, bool)
        f_none, _ = m._obj(th, pi, di, oi, runs, none_cens, 1, 1, 1)
        f_all, _ = m._obj(th, pi, di, oi, runs, all_cens, 1, 1, 1)
        # Treating them as censored assigns higher likelihood at this mean,
        # i.e. censored innings are evidence of *more* ability, not less.
        self.assertLess(f_all, f_none)


class TestEraOffsets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_innings()

    def test_offsets_track_observed_scoring(self):
        off = era_offsets(self.df)
        g = self.df.groupby("decade")
        obs = g.runs.sum() / (g.size() - g.not_out.sum()).clip(lower=1)
        common = off.index.intersection(obs.index)
        r = np.corrcoef(off[common], np.log(obs[common]))[0, 1]
        # By construction this should be essentially 1.0. The free-parameter
        # version scored -0.21, which is what motivated the fixed offset.
        self.assertGreater(r, 0.99)

    def test_early_eras_are_hardest(self):
        off = era_offsets(self.df)
        self.assertLess(off[1880], off[2000])


class TestTools(unittest.TestCase):
    def test_resolves_real_players(self):
        self.assertEqual(T._resolve("Bradman"), "DG Bradman")
        self.assertEqual(T._resolve("Kohli"), "V Kohli")

    def test_resolves_nicknames(self):
        self.assertEqual(T._resolve("Sachin"), "SR Tendulkar")
        self.assertEqual(T._resolve("The Don"), "DG Bradman")

    def test_rejects_junk(self):
        """'Test' once fuzzy-matched onto a real player and corrupted a debate."""
        for junk in ("Test", "Cricket", "xyzzy", "a"):
            self.assertIsNone(T._resolve(junk), f"{junk!r} should not resolve")

    def test_get_player_shape(self):
        p = T.get_player("Bradman")
        self.assertEqual(p["player"], "DG Bradman")
        self.assertEqual(p["cri_plus_rank"], 1)
        self.assertGreater(p["average"], 99)

    def test_unknown_player_returns_error_not_exception(self):
        p = T.get_player("Zzzz Nobody")
        self.assertIn("error", p)
        self.assertIn("suggestions", p)

    def test_everything_is_json_serialisable(self):
        """Agents receive tool output as JSON; numpy types break that."""
        for payload in (
            T.get_player("Tendulkar"),
            T.compare(["Bradman", "Tendulkar"]),
            T.era_context(1930),
            T.leaderboard(5),
            T.search_players("Smith"),
        ):
            json.dumps(payload)

    def test_dispatch_handles_bad_input(self):
        self.assertIn("error", T.dispatch("nope", {}))
        self.assertIn("error", T.dispatch("get_player", {"wrong_kwarg": 1}))

    def test_compare_needs_two_valid_players(self):
        self.assertIn("error", T.compare(["Bradman", "Zzzz Nobody"]))


class TestDebate(unittest.TestCase):
    def setUp(self):
        self.tr = Debate(MockProvider()).run(
            "Who is the greatest batter: Bradman or Tendulkar?"
        )

    def test_all_four_roles_speak(self):
        self.assertEqual(
            [t.role for t in self.tr.turns],
            ["statistician", "historian", "skeptic", "judge"],
        )

    def test_tools_were_called_and_recorded(self):
        self.assertTrue(any(t.tool_calls for t in self.tr.turns))
        self.assertTrue(self.tr.facts)

    def test_verdict_parses(self):
        self.assertIsNotNone(self.tr.verdict)
        self.assertIn("verdict", self.tr.verdict)
        self.assertIsInstance(self.tr.verdict["confidence"], int)

    def test_grounding_guard_catches_invented_number(self):
        """MockProvider injects 87.4, which no tool returns."""
        self.assertIn("87.4", self.tr.ungrounded_numbers())

    def test_clean_verdict_has_no_ungrounded_numbers(self):
        tr = Debate(MockProvider(inject_hallucination=False)).run(
            "Compare Bradman and Tendulkar"
        )
        self.assertNotIn("87.4", tr.ungrounded_numbers())


class TestJsonParsing(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(_parse_json('{"verdict":"x"}')["verdict"], "x")

    def test_fenced(self):
        self.assertEqual(_parse_json('```json\n{"verdict":"x"}\n```')["verdict"], "x")

    def test_embedded_in_prose(self):
        self.assertEqual(_parse_json('Sure!\n{"verdict":"x"}\nHope that helps')["verdict"], "x")

    def test_garbage_degrades_gracefully(self):
        self.assertIn("verdict", _parse_json("not json at all"))




class TestFormats(unittest.TestCase):
    """Multi-format ratings. ODI/T20I data may not be scraped yet."""

    def test_weights_are_documented_and_sum_to_one(self):
        from formats import WEIGHTS

        for fmt, (wd, wt) in WEIGHTS.items():
            self.assertAlmostEqual(wd + wt, 1.0, msg=f"{fmt} weights must sum to 1")

    def test_tempo_matters_more_in_shorter_formats(self):
        from formats import WEIGHTS

        self.assertLess(WEIGHTS["test"][1], WEIGHTS["odi"][1])
        self.assertLess(WEIGHTS["odi"][1], WEIGHTS["t20i"][1])

    def test_missing_format_raises_actionable_error(self):
        from formats import load_format

        with self.assertRaises(FileNotFoundError) as cm:
            load_format("nonexistent_format")
        self.assertIn("fetch_statsguru", str(cm.exception))

    def test_test_format_ignores_strike_rate(self):
        """Tests weight tempo at zero, so BAT+ must equal CRI+ exactly."""
        import pandas as pd
        from formats import PROC

        path = PROC / "cri_plus_test.parquet"
        if not path.exists():
            self.skipTest("test ratings not built yet")
        d = pd.read_parquet(path)
        pd.testing.assert_series_equal(
            d.bat_plus, d.cri_plus, check_names=False, rtol=1e-9
        )

    def test_strike_rate_index_needs_enough_innings(self):
        """SR+ must be NaN, not a guess, where balls-faced data is thin."""
        import numpy as np
        import pandas as pd
        from formats import strike_rate_index

        df = pd.DataFrame({
            "player": ["A"] * 3 + ["B"] * 20,
            "runs": [50] * 23,
            "balls_faced": [100.0] * 23,
            "not_out": [False] * 23,
            "decade": [2010] * 23,
            "opposition": ["India"] * 23,
        })
        idx = strike_rate_index(df, pd.Index(["A", "B"]))
        self.assertTrue(np.isnan(idx["A"]), "3 innings is too few for SR+")
        self.assertFalse(np.isnan(idx["B"]))


class TestBowlPlus(unittest.TestCase):
    """BOWL+ couples two Poisson rate models; the joint gradient must be right."""

    def test_gradient(self):
        from bowl_plus import BowlPlusModel

        rng = np.random.default_rng(3)
        n, nb, nd, no = 400, 10, 4, 5
        bi = rng.integers(0, nb, n)
        di = rng.integers(0, nd, n)
        oi = rng.integers(0, no, n)
        balls = rng.integers(30, 300, n).astype(float)
        wkts = rng.poisson(balls * 0.017).astype(float)
        runs = rng.poisson(balls * 0.5).astype(float)

        m = BowlPlusModel()
        m.era_w_ = rng.normal(0, 0.1, nd)
        m.era_r_ = rng.normal(0, 0.1, nd)
        # Time-varying opposition strength enters both rate models.
        m.oppoff_ = rng.normal(0, 0.15, n)
        th = rng.normal(0, 0.2, 2 * nb + 2 + 2 * no)
        th[2 * nb] = np.log(0.017)
        th[2 * nb + 1] = np.log(0.5)
        args = (bi, di, oi, balls, wkts, runs, nb, nd, no)

        _, g = m._obj(th, *args)
        # The objective is ~1e5, so eps=1e-6 is dominated by float round-off.
        # Richardson extrapolation at eps=1e-4 is the correctly conditioned check.
        eps = 1e-4
        num = np.zeros_like(th)
        for k in range(len(th)):
            d = np.zeros_like(th)
            d[k] = 1.0
            f1 = (m._obj(th + eps * d, *args)[0] - m._obj(th - eps * d, *args)[0]) / (2 * eps)
            f2 = (m._obj(th + 2 * eps * d, *args)[0] - m._obj(th - 2 * eps * d, *args)[0]) / (4 * eps)
            num[k] = (4 * f1 - f2) / 3
        self.assertLess(float(np.max(np.abs(g - num) / (1 + np.abs(num)))), 1e-6)

    def test_missing_data_raises_actionable_error(self):
        from bowl_plus import load_bowling

        with self.assertRaises(FileNotFoundError) as cm:
            load_bowling("nope")
        self.assertIn("fetch_statsguru", str(cm.exception))


class TestAssociateContamination(unittest.TestCase):
    """Guards the T20I fix: associate-vs-associate matches must not leak in."""

    def test_full_members_list_is_the_twelve(self):
        from formats import FULL_MEMBERS

        self.assertEqual(len(FULL_MEMBERS), 12)
        for side in ("India", "Australia", "Zimbabwe", "Ireland", "Afghanistan"):
            self.assertIn(side, FULL_MEMBERS)
        for side in ("Austria", "Bulgaria", "Spain", "Saudi Arabia"):
            self.assertNotIn(side, FULL_MEMBERS)

    def test_loader_filters_opposition(self):
        from formats import FULL_MEMBERS, load_format

        try:
            df = load_format("t20i")
        except FileNotFoundError:
            self.skipTest("t20i data not scraped yet")
        self.assertTrue(df.opposition.isin(FULL_MEMBERS).all())

    def test_no_associate_nations_in_t20i_top_50(self):
        import pandas as pd
        from formats import FULL_MEMBERS, PROC

        path = PROC / "cri_plus_t20i.parquet"
        if not path.exists():
            self.skipTest("t20i ratings not built yet")
        top = pd.read_parquet(path).nlargest(50, "bat_plus")
        # Country codes, not full names, so check a few known offenders.
        for bad in ("AUT", "BUL", "ESP", "KSA"):
            self.assertNotIn(bad, set(top.country), f"{bad} should not rank top-50")

    def test_sr_plus_is_shrunk_toward_100(self):
        """A tiny sample must not produce an extreme tempo index."""
        import numpy as np
        import pandas as pd
        from formats import strike_rate_index

        df = pd.DataFrame({
            "player": ["small"] * 12 + ["big"] * 200,
            "runs": [30] * 12 + [30] * 200,
            "balls_faced": [10.0] * 12 + [10.0] * 200,
            "not_out": [False] * 212,
            "decade": [2020] * 212,
            "opposition": ["India"] * 212,
        })
        idx = strike_rate_index(df, pd.Index(["small", "big"]))
        # Same per-ball rate, but the small sample is pulled closer to 100.
        self.assertLess(abs(idx["small"] - 100), abs(idx["big"] - 100) + 1e-9)


class TestBaselineConsistency(unittest.TestCase):
    """era_context must quote the same baseline the ratings were fitted on.

    It did not: 2020s T20I read 19.19 from unfiltered innings while the ratings
    used 21.83 from Full-Member-only innings. The panel would then reason about
    an era that does not correspond to the numbers it is comparing.
    """

    def test_era_context_matches_rating_baseline(self):
        from formats import load_format

        for fmt, decade in (("test", 1930), ("odi", 1980), ("t20i", 2020)):
            try:
                d = load_format(fmt)
            except FileNotFoundError:
                continue
            d = d[d.year // 10 * 10 == decade]
            if d.empty:
                continue
            expected = d.runs.sum() / max(len(d) - int(d.not_out.sum()), 1)
            got = T.era_context(decade, format=fmt)["batting_average"]
            self.assertAlmostEqual(got, round(expected, 2), places=2,
                                   msg=f"{fmt} {decade}s baseline drifted")

    def test_innings_lookups_are_full_member_only(self):
        from formats import FULL_MEMBERS

        for fmt in ("test", "odi", "t20i"):
            try:
                df = T._innings(fmt)
            except FileNotFoundError:
                continue
            self.assertTrue(df.opposition.isin(FULL_MEMBERS).all(), fmt)


class TestFormatAwareness(unittest.TestCase):
    """Serving Test numbers for an ODI question is the bug the Skeptic caught."""

    def test_ratings_differ_by_format(self):
        try:
            t = T.get_player("Viv", format="test")
            o = T.get_player("Viv", format="odi")
        except FileNotFoundError:
            self.skipTest("odi ratings not built")
        if "error" in t or "error" in o:
            self.skipTest("player not in both formats")
        self.assertEqual(t["format"], "TEST")
        self.assertEqual(o["format"], "ODI")
        self.assertNotAlmostEqual(t["rating"], o["rating"], places=1)

    def test_every_tool_schema_accepts_format(self):
        need = {"get_player", "compare", "era_context", "leaderboard",
                "get_bowler", "bowling_leaderboard"}
        for sch in T.TOOL_SCHEMAS:
            if sch["name"] in need:
                self.assertIn("format", sch["parameters"]["properties"],
                              f"{sch['name']} must accept a format")

    def test_bowling_tools_are_format_aware(self):
        import inspect

        for fn in (T.get_bowler, T.bowling_leaderboard):
            self.assertIn("format", inspect.signature(fn).parameters, fn.__name__)


class TestPeakRatings(unittest.TestCase):
    """Peak windows must land on the periods cricket already agrees on."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        from peak import PROC

        path = PROC / "peak_test.parquet"
        if not path.exists():
            raise unittest.SkipTest("peak ratings not built")
        cls.d = pd.read_parquet(path).set_index("player")

    def _window(self, player):
        if player not in self.d.index:
            self.skipTest(f"{player} not rated")
        r = self.d.loc[player]
        return int(r.peak_start), int(r.peak_end)

    def test_known_peaks_land_in_the_right_window(self):
        # Widely agreed peak periods; each must fall inside the stated range.
        expected = {
            "DG Bradman": (1928, 1938),
            "SPD Smith": (2013, 2019),   # the great modern Test peak
            "RT Ponting": (2002, 2008),
            "IVA Richards": (1975, 1981),
            "GS Sobers": (1957, 1966),
        }
        for player, (lo, hi) in expected.items():
            start, end = self._window(player)
            self.assertTrue(lo <= start and end <= hi + 1,
                            f"{player} peak {start}-{end} outside {lo}-{hi}")

    def test_distinct_peaks_never_fall_below_career(self):
        """Where a peak is a genuine subset of the career, it must exceed it.

        Peaks covering most of a short career are flagged `peak_distinct=False`
        instead: shrinkage is evidence-proportional, so a window with fewer
        innings is pulled harder toward the mean, and such a "peak" is really
        just the career measured with less data.
        """
        d = self.d[self.d.peak_distinct]
        bad = d[d.peak_plus < d.cri_plus - 1e-6]
        self.assertEqual(len(bad), 0, f"{len(bad)} distinct peaks below career")

    def test_non_distinct_peaks_are_short_career_artefacts(self):
        nd = self.d[~self.d.peak_distinct]
        if len(nd) == 0:
            self.skipTest("none flagged")
        # They should be dominated by windows covering most of the career.
        self.assertGreater(nd.peak_share.mean(), self.d[self.d.peak_distinct].peak_share.mean())

    def test_bradman_peaks_highest(self):
        self.assertEqual(self.d.peak_plus.idxmax(), "DG Bradman")

    def test_late_career_improvers_show_a_large_lift(self):
        """Imran Khan's batting genuinely transformed late; the lift must show."""
        if "Imran Khan" not in self.d.index:
            self.skipTest("Imran Khan not rated")
        r = self.d.loc["Imran Khan"]
        self.assertGreater(r.peak_plus - r.cri_plus, 50)
        self.assertGreaterEqual(int(r.peak_start), 1985)


class TestBowlPlusRatings(unittest.TestCase):
    """BOWL+ must recover the bowlers everyone already agrees on."""

    CANON = ["MD Marshall", "GD McGrath", "M Muralidaran", "SK Warne", "DW Steyn",
             "Sir RJ Hadlee", "CEL Ambrose", "Imran Khan", "JJ Bumrah",
             "Wasim Akram", "CA Walsh", "JM Anderson", "R Ashwin", "SF Barnes"]

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        from bowl_plus import PROC

        path = PROC / "bowl_plus_test.parquet"
        if not path.exists():
            raise unittest.SkipTest("bowling ratings not built")
        d = pd.read_parquet(path).sort_values("bowl_plus", ascending=False)
        cls.d = d.reset_index(drop=True)
        cls.d["rank"] = cls.d.index + 1

    def test_canonical_bowlers_rank_highly(self):
        found = [p for p in self.CANON if p in set(self.d.player)]
        top = set(self.d[self.d["rank"] <= 60].player)
        hits = [p for p in found if p in top]
        self.assertGreaterEqual(len(hits) / max(len(found), 1), 0.85,
                                f"only {len(hits)}/{len(found)} greats in top 60")

    def test_intervals_narrow_with_wickets(self):
        d = self.d.assign(w=self.d.bowl_hi - self.d.bowl_lo)
        bands = [(0, 120), (120, 300), (300, 10_000)]
        widths = [d[(d.wickets >= lo) & (d.wickets < hi)].w.mean() for lo, hi in bands]
        widths = [w for w in widths if w == w]
        self.assertTrue(all(a > b for a, b in zip(widths, widths[1:])),
                        f"CI widths not monotone: {widths}")

    def test_small_samples_do_not_get_established_leads(self):
        """An 85-wicket career must not out-rank an 800-wicket one decisively."""
        d = self.d
        big = d[d.wickets >= 400]
        small = d[d.wickets < 120]
        if big.empty or small.empty:
            self.skipTest("bands empty")
        # No sub-120-wicket bowler should clear the best 400+ bowler's interval.
        best_big_hi = big.bowl_hi.max()
        self.assertTrue((small.bowl_lo < best_big_hi).all(),
                        "a small-sample bowler claims an established lead")

    def test_opposition_strength_is_time_varying(self):
        """A single 150-year opposition constant mis-rated 1890s bowlers."""
        from bowl_plus import load_bowling, opp_era_offsets

        off = opp_era_offsets(load_bowling("test"))
        sa = off[off.index.get_level_values("opposition") == "South Africa"]
        sa = sa[sa != 0.0]
        if len(sa) < 3:
            self.skipTest("insufficient South Africa decades")
        # Early South Africa batted far worse than modern South Africa.
        early = sa[sa.index.get_level_values("decade") < 1920]
        late = sa[sa.index.get_level_values("decade") >= 1990]
        if len(early) and len(late):
            self.assertLess(early.mean(), late.mean())


class TestDataIntegrity(unittest.TestCase):
    """Career totals must match the real record exactly.

    This is the sharpest check available: the answers are public and not open
    to interpretation. It has already caught two serious bugs.

    1. Fixed column positions. In 8-ball-over eras Statsguru inserts a BPO
       column, shifting every field right by one, so start_date read the ground
       name, to_datetime produced NaT and dropna deleted the row. 1946-1979
       vanished silently, taking Sobers's 235 wickets with it -- 12,390 innings
       in total. Columns are now mapped by header name.
    2. Name-keyed identity. Two Pakistanis named "Imran Khan" merged into one
       1971-2019 career with 391 wickets, and Richard Hadlee split in two when
       he was knighted mid-career. Identity is now Statsguru's player id.
    """

    KNOWN_TEST_WICKETS = {
        "M Muralidaran": 800, "SK Warne": 708, "JM Anderson": 704,
        "GD McGrath": 563, "CA Walsh": 519, "DW Steyn": 439,
        "N Kapil Dev": 434, "Sir RJ Hadlee": 431, "Wasim Akram": 414,
        "CEL Ambrose": 405, "MD Marshall": 376, "Imran Khan (1971)": 362,
        "DK Lillee": 355, "FS Trueman": 307, "GS Sobers": 235,
        "JC Laker": 193, "SF Barnes": 189,
    }

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        from bowl_plus import PROC

        path = PROC / "bowl_plus_test.parquet"
        if not path.exists():
            raise unittest.SkipTest("bowling ratings not built")
        cls.b = pd.read_parquet(path).set_index("player")

    def test_career_wicket_totals_match_the_record(self):
        missing, wrong = [], []
        for name, wkts in self.KNOWN_TEST_WICKETS.items():
            if name not in self.b.index:
                missing.append(name)
                continue
            got = int(self.b.loc[name, "wickets"])
            if got != wkts:
                wrong.append(f"{name}: got {got}, record says {wkts}")
        self.assertEqual(missing, [], f"absent from the data: {missing}")
        self.assertEqual(wrong, [], f"totals disagree with the record: {wrong}")

    def test_no_decade_is_missing_bowling_data(self):
        """Entire decades once vanished without raising anything."""
        import pandas as pd
        from bowl_plus import PROC

        bat = pd.read_parquet(PROC / "test_innings.parquet")
        bowl = pd.read_parquet(PROC / "test_bowling.parquet")
        b = bat.groupby("year").size()
        w = bowl.groupby("year").size()
        gaps = [int(y) for y in b.index if b[y] > 50 and w.get(y, 0) == 0]
        self.assertEqual(gaps, [], f"years with batting but no bowling: {gaps}")

    def test_eight_ball_overs_are_converted_correctly(self):
        import pandas as pd
        from bowl_plus import PROC

        bowl = pd.read_parquet(PROC / "test_bowling.parquet")
        if "balls_per_over" not in bowl.columns:
            self.skipTest("balls_per_over not captured")
        self.assertGreater((bowl.balls_per_over == 8).sum(), 1000,
                           "8-ball-over innings should be present")
        # Balls must always be consistent with the stated over length.
        eight = bowl[bowl.balls_per_over == 8]
        self.assertTrue((eight.balls % 8 <= 7).all())

    def test_player_ids_are_the_identity(self):
        import pandas as pd
        from bowl_plus import PROC

        d = pd.read_parquet(PROC / "test_bowling.parquet")
        self.assertIn("player_id", d.columns)
        self.assertGreater(d.player_id.notna().mean(), 0.99)


class TestRuleChanges(unittest.TestCase):
    """Rule-change effects must always carry an interval, never a bare number."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        from rulechange import PROC

        path = PROC / "rule_changes.parquet"
        if not path.exists():
            raise unittest.SkipTest("rule-change analysis not built")
        cls.d = pd.read_parquet(path)

    def test_every_estimate_has_an_interval(self):
        self.assertTrue((self.d.ci_low <= self.d["diff"]).all())
        self.assertTrue((self.d["diff"] <= self.d.ci_high).all())

    def test_significance_requires_both_criteria(self):
        """Interval excluding zero AND p<0.05 — noise must not read as effect."""
        for _, r in self.d.iterrows():
            excl = (r.ci_low > 0) or (r.ci_high < 0)
            self.assertEqual(bool(r.significant), bool(excl and r.p_value < 0.05),
                             f"{r.change}: significance flag disagrees with its own stats")

    def test_known_effects_are_recovered(self):
        """Two new balls (2011) and the 2012 circle rule both raised ODI scoring."""
        for key in ("odi_two_balls", "odi_four_out"):
            r = self.d[self.d.change == key]
            if r.empty:
                continue
            r = r.iloc[0]
            self.assertGreater(r["diff"], 0, f"{key} should raise scoring")
            self.assertTrue(r.significant, f"{key} should be significant")

    def test_both_estimators_are_reported(self):
        """They can disagree, and the disagreement is the informative part."""
        have_its = self.d.its_effect.notna()
        self.assertGreater(have_its.sum(), 0)

    def test_tool_output_is_json_safe(self):
        import json

        json.dumps(T.rule_change_effect(format="odi"))


class TestFormatAwareThresholds(unittest.TestCase):
    """Test-sized workload bars silently produced empty frames and NaNs.

    A T20I bowler delivers 24 balls a match, so a 3000-ball threshold qualified
    nobody. Every workload bar now scales with the format.
    """

    def test_bowling_thresholds_scale_down(self):
        from bowl_plus import MIN_BALLS_BY_FORMAT as M

        self.assertGreater(M["test"], M["odi"])
        self.assertGreater(M["odi"], M["t20i"])

    def test_allrounder_thresholds_scale_down(self):
        from allrounder import QUAL

        self.assertGreater(QUAL["test"][1], QUAL["odi"][1])
        self.assertGreater(QUAL["odi"][1], QUAL["t20i"][1])

    def test_empty_qualification_raises_rather_than_returning_nan(self):
        from bowl_plus import build

        with self.assertRaises((ValueError, FileNotFoundError)):
            build("t20i", min_balls=10_000_000)

    def test_every_format_has_every_artifact(self):
        from cri_plus import PROC

        missing = []
        for fmt in ("test", "odi", "t20i"):
            for art in ("cri_plus", "bowl_plus", "peak", "all_plus"):
                if not (PROC / f"{art}_{fmt}.parquet").exists():
                    missing.append(f"{art}_{fmt}")
        self.assertEqual(missing, [], f"missing artifacts: {missing}")


class TestPeakAcrossFormats(unittest.TestCase):
    """Peak windows must land on the right period in every format."""

    KNOWN = {
        "odi": {"V Kohli": (2015, 2020), "IVA Richards": (1979, 1987)},
        "test": {"DG Bradman": (1928, 1938), "SPD Smith": (2013, 2019)},
    }

    def test_known_peaks(self):
        import pandas as pd
        from peak import PROC

        for fmt, expect in self.KNOWN.items():
            path = PROC / f"peak_{fmt}.parquet"
            if not path.exists():
                continue
            d = pd.read_parquet(path).set_index("player")
            for player, (lo, hi) in expect.items():
                if player not in d.index:
                    continue
                r = d.loc[player]
                self.assertTrue(lo <= int(r.peak_start) and int(r.peak_end) <= hi + 1,
                                f"{fmt} {player}: {int(r.peak_start)}-{int(r.peak_end)} "
                                f"outside {lo}-{hi}")


class TestWomensCricket(unittest.TestCase):
    """Women's cricket is rated on the same footing as the men's game."""

    def test_team_names_normalise_to_one_vocabulary(self):
        from formats import FULL_MEMBERS, normalise_team

        for raw, want in [("NZ Women", "New Zealand"), ("SL Women", "Sri Lanka"),
                          ("England Women", "England"), ("WI Women", "West Indies"),
                          ("SA Women", "South Africa")]:
            self.assertEqual(normalise_team(raw), want)
            self.assertIn(normalise_team(raw), FULL_MEMBERS)

    def test_normalisation_is_idempotent_for_mens_names(self):
        from formats import normalise_team

        for n in ("Australia", "South Africa", "West Indies", "India"):
            self.assertEqual(normalise_team(n), n)
            self.assertEqual(normalise_team(normalise_team(n)), n)

    def test_country_codes_stay_abbreviated_in_both_games(self):
        """Expanding country codes made the same nation render two ways."""
        from formats import normalise_country

        for raw, want in [("NZ-W", "NZ"), ("SL-W", "SL"), ("ENG-W", "ENG"),
                          ("AUS", "AUS")]:
            self.assertEqual(normalise_country(raw), want)

    def test_womens_odi_ratings_match_consensus(self):
        """Lanning, Rolton, Raj and Clark are not controversial picks."""
        import pandas as pd
        from formats import PROC

        path = PROC / "cri_plus_wodi.parquet"
        if not path.exists():
            self.skipTest("women's ODI ratings not built")
        d = pd.read_parquet(path).sort_values("bat_plus", ascending=False)
        top15 = set(d.head(15).player)
        for p in ("MM Lanning", "KL Rolton", "M Raj", "BJ Clark", "EA Perry"):
            self.assertIn(p, top15, f"{p} should rank top-15 in women's ODIs")
        self.assertEqual(d.iloc[0].player, "MM Lanning")

    def test_associates_still_excluded_after_normalising(self):
        from formats import FULL_MEMBERS, normalise_team

        for n in ("Thailand Women", "Netherlands Women", "Scotland Women"):
            self.assertNotIn(normalise_team(n), FULL_MEMBERS)

    def test_womens_formats_are_registered(self):
        from fetch_statsguru import FORMATS
        from formats import MIN_INNINGS, WEIGHTS, is_womens

        for f in ("wtest", "wodi", "wt20i"):
            self.assertIn(f, FORMATS)
            self.assertIn(f, WEIGHTS)
            self.assertIn(f, MIN_INNINGS)
            self.assertTrue(is_womens(f))

    def test_womens_test_bar_is_lower(self):
        """Barely 150 women's Tests have ever been played."""
        from formats import MIN_INNINGS

        self.assertLess(MIN_INNINGS["wtest"], MIN_INNINGS["test"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
