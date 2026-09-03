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
        # Whatever's actually next in the real BACKLOG.md right now.
        self.assertEqual(task, agent_run.next_backlog_item())


class TestVerify(unittest.TestCase):
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
        try:
            passed, log = agent_run.verify(run=self._fake_run_all_pass(good_html))
            self.assertTrue(passed)
        finally:
            (agent_run.ROOT / "out" / "index.html").unlink(missing_ok=True)

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
        try:
            passed, log = agent_run.verify(run=self._fake_run_all_pass(thin_html))
            self.assertFalse(passed)
            self.assertIn("missing=", log)
        finally:
            (agent_run.ROOT / "out" / "index.html").unlink(missing_ok=True)

    def test_fails_when_site_suspiciously_small(self):
        tiny_html = ("Settle Test batting ODI batting T20I batting "
                     "Women's ODI Women's T20I Women's Test "
                     "Test bowling ODI bowling T20I bowling")
        try:
            passed, log = agent_run.verify(run=self._fake_run_all_pass(tiny_html))
            self.assertFalse(passed)
        finally:
            (agent_run.ROOT / "out" / "index.html").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
