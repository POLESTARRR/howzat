"""Tests for the autonomous-build agent (scripts/agent_tools.py,
scripts/agent_run.py).

Run: PYTHONPATH=src python3 -m unittest discover -s tests -v

No real network calls: the tool-loop tests use FakeGateway, a scripted
stand-in shaped like src/mock_provider.py's MockProvider.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import agent_tools  # noqa: E402
import agent_run  # noqa: E402


class FakeGateway:
    """Scripted stand-in for src.gateway.Gateway, shaped the same way
    (generate/text/calls) so run_agent_loop can't tell the difference.
    `script` is a list of responses, one consumed per .generate() call.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls_made = 0

    def generate(self, prompt, *, tier="cheap", system=None, tools=None, history=None, **_):
        self.calls_made += 1
        if not self.script:
            raise AssertionError("FakeGateway script exhausted")
        return self.script.pop(0)

    @staticmethod
    def text(resp):
        parts = resp.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip()

    @staticmethod
    def calls(resp):
        parts = resp.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return [p["functionCall"] for p in parts if "functionCall" in p]


def _fc(name, args):
    return {"candidates": [{"content": {"parts": [{"functionCall": {"name": name, "args": args}}]}}]}


def _txt(s):
    return {"candidates": [{"content": {"parts": [{"text": s}]}}]}


class TestPathSafety(unittest.TestCase):
    def test_read_file_reads_relative_to_repo_root(self):
        content = agent_tools.read_file("BACKLOG.md")
        self.assertIn("Howzat backlog", content)

    def test_read_file_missing_raises(self):
        with self.assertRaises(agent_tools.ToolError):
            agent_tools.read_file("no/such/file.md")

    def test_write_file_then_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_root = agent_tools.REPO_ROOT
            agent_tools.REPO_ROOT = Path(tmp)
            try:
                agent_tools.write_file("scratch/note.txt", "hello")
                self.assertEqual(agent_tools.read_file("scratch/note.txt"), "hello")
            finally:
                agent_tools.REPO_ROOT = old_root

    def test_path_cannot_escape_repo_root(self):
        with self.assertRaises(agent_tools.ToolError):
            agent_tools._safe_path("../../etc/passwd")

    def test_absolute_path_outside_root_rejected(self):
        with self.assertRaises(agent_tools.ToolError):
            agent_tools._safe_path("/etc/passwd")


class TestRunCommand(unittest.TestCase):
    def test_captures_stdout_and_exit_code(self):
        out = agent_tools.run_command("echo hello")
        self.assertEqual(out["exit_code"], 0)
        self.assertIn("hello", out["stdout"])

    def test_nonzero_exit_captured_not_raised(self):
        out = agent_tools.run_command("exit 3")
        self.assertEqual(out["exit_code"], 3)

    def test_timeout_reports_as_dict_not_exception(self):
        out = agent_tools.run_command("sleep 5", timeout=1)
        self.assertEqual(out["exit_code"], -1)
        self.assertIn("timed out", out["stderr"])

    def test_runs_with_repo_root_as_cwd(self):
        out = agent_tools.run_command("cat BACKLOG.md")
        self.assertIn("Howzat backlog", out["stdout"])


class TestDispatch(unittest.TestCase):
    def test_dispatch_wraps_success_in_result(self):
        out = agent_tools.dispatch("read_file", {"path": "BACKLOG.md"})
        self.assertIn("Howzat backlog", out["result"])

    def test_dispatch_unknown_tool(self):
        out = agent_tools.dispatch("nonexistent", {})
        self.assertIn("error", out)
        self.assertIn("unknown tool", out["error"])

    def test_dispatch_bad_arguments(self):
        out = agent_tools.dispatch("read_file", {"wrong_kwarg": "x"})
        self.assertIn("error", out)

    def test_dispatch_tool_error_becomes_error_key_not_exception(self):
        out = agent_tools.dispatch("read_file", {"path": "../../etc/passwd"})
        self.assertIn("error", out)

    def test_finish_task_returns_summary(self):
        out = agent_tools.dispatch("finish_task", {"summary": "did the thing"})
        self.assertEqual(out["result"], "did the thing")

    def test_tool_schemas_cover_every_tool(self):
        names = {s["name"] for s in agent_tools.TOOL_SCHEMAS}
        self.assertEqual(names, {"read_file", "write_file", "run_command", "finish_task"})


class TestNextBacklogItem(unittest.TestCase):
    SAMPLE = """# Howzat backlog

## Done
- [x] Something already done

## Next
1. [DONE] BOWL+ all three formats.
2. Home/away split as a model term.
3. Batting-position covariate.

## Known limitations
- Not a bug, just a limit.
"""

    def test_returns_first_unchecked_item(self):
        item = agent_run.next_backlog_item(self.SAMPLE)
        self.assertEqual(item, "Home/away split as a model term.")

    def test_skips_done_items(self):
        text = self.SAMPLE.replace(
            "2. Home/away split as a model term.",
            "2. [DONE] Home/away split as a model term.",
        )
        item = agent_run.next_backlog_item(text)
        self.assertEqual(item, "Batting-position covariate.")

    def test_returns_none_when_all_done(self):
        text = self.SAMPLE.replace(
            "2. Home/away split as a model term.",
            "2. [DONE] Home/away split as a model term.",
        ).replace(
            "3. Batting-position covariate.",
            "3. [DONE] Batting-position covariate.",
        )
        self.assertIsNone(agent_run.next_backlog_item(text))

    def test_returns_none_with_no_next_heading(self):
        self.assertIsNone(agent_run.next_backlog_item("# Just a title\n\nNo sections here.\n"))

    def test_stops_at_next_heading(self):
        text = self.SAMPLE  # "## Known limitations" follows "## Next"
        item = agent_run.next_backlog_item(text)
        self.assertNotIn("Not a bug", item or "")

    def test_reads_real_backlog_file_without_crashing(self):
        # No assertion on content (it changes over time) -- just that the
        # real file parses without raising.
        agent_run.next_backlog_item()


class TestSelectTask(unittest.TestCase):
    def test_failing_tests_take_priority(self):
        calls = []

        def fake_run(command, timeout=300):
            calls.append(command)
            if "unittest discover" in command:
                return {"exit_code": 1, "stdout": "FAILED test_x", "stderr": ""}
            return {"exit_code": 0, "stdout": "", "stderr": ""}

        task = agent_run.select_task(run=fake_run)
        self.assertIn("Fix this failing test suite", task)
        self.assertIn("FAILED test_x", task)
        # validate.py should never even run if the test suite already failed.
        self.assertEqual(len(calls), 1)

    def test_failing_validate_when_tests_pass(self):
        def fake_run(command, timeout=300):
            if "unittest discover" in command:
                return {"exit_code": 0, "stdout": "", "stderr": ""}
            if "validate.py" in command:
                return {"exit_code": 1, "stdout": "3/8 checks passed", "stderr": ""}
            raise AssertionError(f"unexpected command: {command}")

        task = agent_run.select_task(run=fake_run)
        self.assertIn("Fix this failing validation check", task)
        self.assertIn("3/8 checks passed", task)

    def test_falls_back_to_backlog_when_checks_pass(self):
        def fake_run(command, timeout=300):
            return {"exit_code": 0, "stdout": "", "stderr": ""}

        task = agent_run.select_task(run=fake_run)
        # Whatever's actually next in the real BACKLOG.md right now, plus
        # the appended instruction to mark it [DONE] in BACKLOG.md.
        self.assertIn(agent_run.next_backlog_item(), task)
        self.assertIn("mark this item [DONE]", task)

    def test_returns_none_when_backlog_has_nothing_next(self):
        def fake_run(command, timeout=300):
            return {"exit_code": 0, "stdout": "", "stderr": ""}

        old_item = agent_run.next_backlog_item
        agent_run.next_backlog_item = lambda: None
        try:
            self.assertIsNone(agent_run.select_task(run=fake_run))
        finally:
            agent_run.next_backlog_item = old_item


class TestVerify(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_root = agent_run.ROOT
        agent_run.ROOT = Path(self._tmpdir.name)

    def tearDown(self):
        agent_run.ROOT = self._old_root
        self._tmpdir.cleanup()

    def _fake_run_all_pass(self, site_html: str):
        def fake_run(command, timeout=300):
            if "build_site.py" in command:
                (agent_run.ROOT / "out").mkdir(exist_ok=True)
                (agent_run.ROOT / "out" / "index.html").write_text(site_html)
            return {"exit_code": 0, "stdout": "ok", "stderr": ""}
        return fake_run

    def test_passes_when_everything_green(self):
        good_html = ("Settle Test batting ODI batting T20I batting "
                     "Women's ODI Women's T20I Women's Test "
                     "Test bowling ODI bowling T20I bowling ") + ("x" * 300_000)
        passed, log = agent_run.verify(run=self._fake_run_all_pass(good_html))
        self.assertTrue(passed)

    def test_fails_on_first_failing_command_without_running_later_ones(self):
        calls = []

        def fake_run(command, timeout=300):
            calls.append(command)
            if "unittest discover" in command:
                return {"exit_code": 1, "stdout": "boom", "stderr": ""}
            return {"exit_code": 0, "stdout": "", "stderr": ""}

        passed, log = agent_run.verify(run=fake_run)
        self.assertFalse(passed)
        self.assertIn("boom", log)
        self.assertEqual(len(calls), 1)  # never got to validate.py or build_site.py

    def test_fails_when_site_missing_required_tables(self):
        thin_html = "Settle Test batting" + ("x" * 300_000)  # missing most tables
        passed, log = agent_run.verify(run=self._fake_run_all_pass(thin_html))
        self.assertFalse(passed)
        self.assertIn("missing required tables=", log)
        self.assertNotIn("too small", log)

    def test_fails_when_site_suspiciously_small(self):
        tiny_html = ("Settle Test batting ODI batting T20I batting "
                     "Women's ODI Women's T20I Women's Test "
                     "Test bowling ODI bowling T20I bowling")
        passed, log = agent_run.verify(run=self._fake_run_all_pass(tiny_html))
        self.assertFalse(passed)
        self.assertIn("site too small", log)
        self.assertNotIn("missing required tables=", log)


class TestRunAgentLoop(unittest.TestCase):
    def test_finishes_and_verifies_clean(self):
        gw = FakeGateway([
            _fc("read_file", {"path": "BACKLOG.md"}),
            _fc("finish_task", {"summary": "done"}),
        ])
        verified, summary = agent_run.run_agent_loop(
            "a task", gw, verify=lambda: (True, "all green"),
        )
        self.assertTrue(verified)
        self.assertEqual(summary, "done")

    def test_stalls_after_two_textual_non_tool_turns(self):
        gw = FakeGateway([_txt("thinking..."), _txt("still thinking...")])
        verified, reason = agent_run.run_agent_loop(
            "a task", gw, verify=lambda: (True, ""),
        )
        self.assertFalse(verified)
        self.assertIn("stalled", reason)

    def test_turn_budget_exhausted(self):
        gw = FakeGateway([_fc("read_file", {"path": "BACKLOG.md"})] * 3)
        verified, reason = agent_run.run_agent_loop(
            "a task", gw, verify=lambda: (True, ""), max_turns=3,
        )
        self.assertFalse(verified)
        self.assertIn("turn budget exhausted", reason)

    def test_repairs_on_failed_verify_then_succeeds(self):
        gw = FakeGateway([
            _fc("finish_task", {"summary": "first attempt"}),
            _fc("write_file", {"path": "x.py", "content": "fixed"}),
            _fc("finish_task", {"summary": "fixed it"}),
        ])
        verify_results = iter([(False, "tests still failing"), (True, "green")])
        with tempfile.TemporaryDirectory() as tmp:
            old_root = agent_tools.REPO_ROOT
            agent_tools.REPO_ROOT = Path(tmp)
            try:
                verified, summary = agent_run.run_agent_loop(
                    "a task", gw, verify=lambda: next(verify_results), max_repair_turns=3,
                )
            finally:
                agent_tools.REPO_ROOT = old_root
        self.assertTrue(verified)
        self.assertEqual(summary, "fixed it")

    def test_gives_up_after_repair_budget_exhausted(self):
        gw = FakeGateway([
            _fc("finish_task", {"summary": "attempt 1"}),
            _fc("finish_task", {"summary": "attempt 2"}),
        ])
        verified, reason = agent_run.run_agent_loop(
            "a task", gw, verify=lambda: (False, "still red"), max_repair_turns=1,
        )
        self.assertFalse(verified)
        self.assertIn("still red", reason)


class TestSlugify(unittest.TestCase):
    def test_lowercases_and_hyphenates(self):
        self.assertEqual(
            agent_run._slugify("Home/away split as a model term."),
            "home-away-split-as-a-model-term",
        )

    def test_truncates_long_text(self):
        long_task = "a " * 100
        self.assertLessEqual(len(agent_run._slugify(long_task)), 40)

    def test_empty_text_falls_back_to_task(self):
        self.assertEqual(agent_run._slugify("!!!"), "task")


class TestMainSkipsLandingOnUnchangedTree(unittest.TestCase):
    """main() must not call land_change() when run_agent_loop() reports
    verified=True but the working tree has no actual changes (e.g. the
    model called finish_task without writing anything, and verify()
    passed because the tree was already in a passing state).
    """

    def test_unchanged_tree_after_verified_true_skips_land_change(self):
        import subprocess as _subprocess
        import types

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _subprocess.run(["git", "init", "-q"], cwd=root, check=True)

            fake_gateway_module = types.ModuleType("gateway")
            fake_gateway_module.Gateway = lambda: object()
            old_gateway_module = sys.modules.get("gateway")

            old_root = agent_run.ROOT
            old_select_task = agent_run.select_task
            old_run_agent_loop = agent_run.run_agent_loop
            old_land_change = agent_run.land_change

            def fail_if_called(*args, **kwargs):
                raise AssertionError("land_change must not run on an unchanged tree")

            agent_run.ROOT = root
            sys.modules["gateway"] = fake_gateway_module
            agent_run.select_task = lambda: "a task"
            agent_run.run_agent_loop = lambda task, gateway: (True, "already satisfied")
            agent_run.land_change = fail_if_called

            try:
                result = agent_run.main()
            finally:
                agent_run.ROOT = old_root
                agent_run.select_task = old_select_task
                agent_run.run_agent_loop = old_run_agent_loop
                agent_run.land_change = old_land_change
                if old_gateway_module is None:
                    sys.modules.pop("gateway", None)
                else:
                    sys.modules["gateway"] = old_gateway_module

            self.assertEqual(result, 0)


class TestWriteSummary(unittest.TestCase):
    def test_writes_to_github_step_summary_when_set(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summary.md"
            old = os.environ.get("GITHUB_STEP_SUMMARY")
            os.environ["GITHUB_STEP_SUMMARY"] = str(summary_path)
            try:
                agent_run._write_summary("hello from the agent")
                self.assertEqual(summary_path.read_text(), "hello from the agent")
            finally:
                if old is None:
                    os.environ.pop("GITHUB_STEP_SUMMARY", None)
                else:
                    os.environ["GITHUB_STEP_SUMMARY"] = old

    def test_does_not_crash_when_unset(self):
        import os
        old = os.environ.pop("GITHUB_STEP_SUMMARY", None)
        try:
            agent_run._write_summary("no crash please")  # just must not raise
        finally:
            if old is not None:
                os.environ["GITHUB_STEP_SUMMARY"] = old


if __name__ == "__main__":
    unittest.main()
