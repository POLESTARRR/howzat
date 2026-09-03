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


if __name__ == "__main__":
    unittest.main()
