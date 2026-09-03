# Autonomous Build Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the "Howzat — continuous build" Claude Code cloud routine with a free GitHub Actions workflow that keeps doing autonomous bug-fix/backlog work on this repo after the Claude subscription ends.

**Architecture:** A cron-scheduled GitHub Actions job runs `scripts/agent_run.py`, which picks one deterministic task (a failing check, else the next BACKLOG.md item), runs a bounded tool-calling loop against the project's existing `src/gateway.py` Gemini client, verifies the result with the same checks `ci.yml` already runs, and — only if they pass — opens a PR that auto-merges once GitHub's own independent CI run on that PR also passes.

**Tech Stack:** Python 3.12 (stdlib only for the new code — `src/gateway.py` already handles the Gemini HTTP calls via `requests`, already a project dependency), GitHub Actions, `gh` CLI (preinstalled on Actions runners).

**Spec:** `specs/2026-09-03-autonomous-build-agent-design.md`

## Global Constraints

- Whole-file writes only, never diffs/patches (spec rationale: more reliable for a smaller model).
- Land changes via PR + `gh pr merge --auto`, never a direct push to `main`.
- Deterministic task selection only: failing check, else next unchecked BACKLOG.md `## Next` item. No open-ended bug-hunting, no IDEAS.md brainstorming.
- Cron cadence: every 6 hours (`0 */6 * * *`).
- Gemini tier: `"strong"` (accuracy bar justifies it over `"cheap"` — see spec).
- Turn budget: 25 turns hard cap per run; up to 3 repair attempts after a failed verify, drawn from the same budget.
- Git commits/PRs: identity `Dhruv Sharma <274071256+POLESTARRR@users.noreply.github.com>`, matching every existing commit. No AI attribution, no `Co-Authored-By` trailer, anywhere — commit messages, PR bodies.
- New files only live in `scripts/` and `tests/` — nothing touches `docs/` (GitHub Pages publish folder, wholesale overwritten by `src/build_site.py`; confirmed by reading its source).

---

## File Structure

- Create: `scripts/agent_tools.py` — tool implementations the model can call (`read_file`, `write_file`, `run_command`, `finish_task`), their Gemini function-call schemas (`TOOL_SCHEMAS`), and a `dispatch()` that mirrors `src/tools.py`'s existing convention exactly (`{"result": ...}` / `{"error": ...}`, never raising into the caller).
- Create: `scripts/agent_run.py` — the orchestrator: `next_backlog_item()`, `select_task()`, `run_agent_loop()`, `verify()`, `land_change()`, `main()`.
- Create: `tests/test_agent_run.py` — covers both new modules, same `unittest` style as `tests/test_howzat.py`, against a `FakeGateway` (mirrors `src/mock_provider.py`'s shape) so no real API calls happen in CI.
- Create: `.github/workflows/autonomous-build.yml` — the cron workflow.

**Interfaces summary** (so later tasks agree on exact names/types):
- `agent_tools.dispatch(name: str, args: dict) -> dict` — `{"result": Any}` or `{"error": str}`.
- `agent_tools.run_command(command: str, timeout: int = 300) -> dict` — `{"exit_code": int, "stdout": str, "stderr": str}` (used directly by `agent_run.py`, not through `dispatch`, since it's an internal caller not a model tool call).
- `agent_tools.TOOL_SCHEMAS: list[dict]` — Gemini function-declaration format.
- `agent_run.next_backlog_item(text: str | None = None) -> str | None`
- `agent_run.select_task() -> str | None`
- `agent_run.run_agent_loop(task: str, gateway, verify=verify, max_turns=25, max_repair_turns=3) -> tuple[bool, str]` — `(verified, summary_or_reason)`.
- `agent_run.verify(run=agent_tools.run_command) -> tuple[bool, str]` — `(passed, log)`.
- `agent_run._slugify(text: str) -> str`
- `agent_run.land_change(task: str, summary: str) -> str` — returns PR URL.

---

### Task 1: `agent_tools.py` — path safety, `read_file`, `write_file`

**Files:**
- Create: `scripts/agent_tools.py`
- Test: `tests/test_agent_run.py`

**Interfaces:**
- Produces: `agent_tools.REPO_ROOT: Path`, `agent_tools.ToolError`, `agent_tools._safe_path(path: str) -> Path`, `agent_tools.read_file(path: str) -> str`, `agent_tools.write_file(path: str, content: str) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent_run.py` with this content:

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'agent_tools'` (the module doesn't exist yet).

- [ ] **Step 3: Create `scripts/agent_tools.py` with the path-safety and file tools**

```python
"""Tools the autonomous-build agent can call: read/write files, run shell
commands, and signal it's done. Mirrors src/tools.py's TOOL_SCHEMAS +
dispatch() convention so this speaks the same shape as the rest of the
project's Gemini integration.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class ToolError(Exception):
    """A tool call that can't be satisfied — bad path, bad args, etc."""


def _safe_path(path: str) -> Path:
    """Resolve `path` against REPO_ROOT and refuse anything that escapes it.

    Runs inside a disposable GitHub Actions container, but the model still
    shouldn't be able to read or write outside the checked-out repo — an
    absolute path or a `../` climb is a mistake to catch, not a real attack
    to defend against in this context.
    """
    candidate = (REPO_ROOT / path).resolve()
    root = REPO_ROOT.resolve()
    if candidate != root and root not in candidate.parents:
        raise ToolError(f"path escapes the repo root: {path!r}")
    return candidate


def read_file(path: str) -> str:
    p = _safe_path(path)
    if not p.is_file():
        raise ToolError(f"no such file: {path}")
    return p.read_text()


def write_file(path: str, content: str) -> str:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} bytes to {path}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
cd ~/Desktop/CRICKET/howzat
git add scripts/agent_tools.py tests/test_agent_run.py
git commit -m "Add path-safe read_file/write_file tools for the autonomous build agent"
```

---

### Task 2: `agent_tools.py` — `run_command`

**Files:**
- Modify: `scripts/agent_tools.py`
- Modify: `tests/test_agent_run.py`

**Interfaces:**
- Produces: `agent_tools.run_command(command: str, timeout: int = 300) -> dict` — `{"exit_code": int, "stdout": str, "stderr": str}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_run.py` (new class, keep existing ones):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run.TestRunCommand -v`
Expected: FAIL — `AttributeError: module 'agent_tools' has no attribute 'run_command'`

- [ ] **Step 3: Add `run_command` to `scripts/agent_tools.py`**

```python
def run_command(command: str, timeout: int = 300) -> dict:
    """Runs `command` in a shell, cwd pinned to the repo root.

    Never raises for a failing or slow command — a bad exit code or a
    timeout is data the agent loop needs to see and react to, not a crash.
    """
    try:
        proc = subprocess.run(
            command, shell=True, cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": f"command timed out after {timeout}s"}
    # Tool results get embedded in the next prompt; keep them bounded.
    return {"exit_code": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/agent_tools.py tests/test_agent_run.py
git commit -m "Add run_command tool with timeout handling"
```

---

### Task 3: `agent_tools.py` — `finish_task`, `TOOL_SCHEMAS`, `dispatch`

**Files:**
- Modify: `scripts/agent_tools.py`
- Modify: `tests/test_agent_run.py`

**Interfaces:**
- Produces: `agent_tools.finish_task(summary: str) -> str`, `agent_tools.TOOL_SCHEMAS: list[dict]`, `agent_tools.TOOLS: dict[str, Callable]`, `agent_tools.dispatch(name: str, args: dict) -> dict`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_run.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run.TestDispatch -v`
Expected: FAIL — `AttributeError: module 'agent_tools' has no attribute 'dispatch'`

- [ ] **Step 3: Add `finish_task`, `TOOL_SCHEMAS`, `TOOLS`, `dispatch` to `scripts/agent_tools.py`**

```python
def finish_task(summary: str) -> str:
    """The model calls this to end its turn loop. agent_run.py's loop
    watches for this specific tool name; this function just echoes the
    summary back so dispatch()'s uniform {"result": ...} shape still holds.
    """
    return summary


_STR = {"type": "string"}
_INT = {"type": "integer"}

TOOL_SCHEMAS = [
    {
        "name": "read_file",
        "description": "Read a file's contents. Path is relative to the repo root.",
        "parameters": {
            "type": "object",
            "properties": {"path": _STR},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Overwrite a file's ENTIRE contents with `content`. Creates the "
            "file (and any parent directories) if it doesn't exist. Always "
            "pass the file's complete new text, never a diff or a partial "
            "snippet -- anything you don't include is deleted."
        ),
        "parameters": {
            "type": "object",
            "properties": {"path": _STR, "content": _STR},
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command in the repo root, e.g. to run tests or "
            "inspect a file. Returns exit code, stdout and stderr."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": _STR,
                "timeout": dict(_INT, description="Seconds, default 300"),
            },
            "required": ["command"],
        },
    },
    {
        "name": "finish_task",
        "description": (
            "Call this exactly once, when the task is complete or you are "
            "stopping. Ending the run any other way (plain text, silence) "
            "does not work -- you must call this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": dict(
                    _STR,
                    description="One or two sentences: what changed and why, or why you stopped.",
                ),
            },
            "required": ["summary"],
        },
    },
]

TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
    "finish_task": finish_task,
}


def dispatch(name: str, args: dict) -> dict:
    """Runs one tool call from the model, never raising into the loop.

    Mirrors src/tools.py's dispatch() exactly: {"result": ...} on success,
    {"error": ...} on any failure, so a bad call becomes something the model
    can read and react to next turn instead of crashing the run.
    """
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}", "available": list(TOOLS)}
    try:
        return {"result": fn(**args)}
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    except ToolError as e:
        return {"error": str(e)}
    except Exception as e:  # noqa: BLE001 - a broken tool must not kill the run
        return {"error": f"{type(e).__name__}: {e}"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run -v`
Expected: PASS (15 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/agent_tools.py tests/test_agent_run.py
git commit -m "Add finish_task, TOOL_SCHEMAS and dispatch() for the agent's tools"
```

---

### Task 4: `agent_run.py` — `next_backlog_item()`

**Files:**
- Create: `scripts/agent_run.py`
- Modify: `tests/test_agent_run.py`

**Interfaces:**
- Consumes: nothing yet from earlier tasks.
- Produces: `agent_run.next_backlog_item(text: str | None = None) -> str | None`, `agent_run.ROOT: Path`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_run.py` (also add the import near the top, next to `import agent_tools`):

```python
import agent_run  # noqa: E402
```

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run.TestNextBacklogItem -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_run'`

- [ ] **Step 3: Create `scripts/agent_run.py` with `next_backlog_item()`**

```python
"""Orchestrates one autonomous-build run: pick a task, let the model work it
with tools, verify the result, and land it as a PR.

Replaces the "Howzat -- continuous build" Claude Code cloud routine with a
free path that keeps working after the Claude subscription ends: this talks
to Gemini through the project's own src/gateway.py, never Claude.

Run directly: PYTHONPATH=src python3 scripts/agent_run.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import agent_tools  # noqa: E402

_NEXT_ITEM_RE = re.compile(r"^\d+\.\s+(.*)$")


def next_backlog_item(text: str | None = None) -> str | None:
    """First unchecked line under BACKLOG.md's '## Next' heading, or None
    if the heading is missing or every item under it is [DONE].
    """
    text = text if text is not None else (ROOT / "BACKLOG.md").read_text()
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "## Next") + 1
    except StopIteration:
        return None
    for line in lines[start:]:
        if line.strip().startswith("## "):
            break
        match = _NEXT_ITEM_RE.match(line.strip())
        if not match:
            continue
        item = match.group(1)
        if item.startswith("[DONE]"):
            continue
        return item
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run -v`
Expected: PASS (21 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/agent_run.py tests/test_agent_run.py
git commit -m "Add BACKLOG.md ## Next parsing for deterministic task selection"
```

---

### Task 5: `agent_run.py` — `select_task()`

**Files:**
- Modify: `scripts/agent_run.py`
- Modify: `tests/test_agent_run.py`

**Interfaces:**
- Consumes: `agent_tools.run_command` (Task 2), `next_backlog_item` (Task 4).
- Produces: `agent_run.select_task(run=agent_tools.run_command) -> str | None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_run.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run.TestSelectTask -v`
Expected: FAIL — `AttributeError: module 'agent_run' has no attribute 'select_task'`

- [ ] **Step 3: Add `select_task()` to `scripts/agent_run.py`**

```python
def select_task(run=agent_tools.run_command) -> str | None:
    """A failing check first, else the next unchecked BACKLOG.md item.

    `run` is injected so tests don't have to run the real (slow) suite --
    defaults to the real agent_tools.run_command for actual use.
    """
    tests = run("PYTHONPATH=src python3 -m unittest discover -s tests")
    if tests["exit_code"] != 0:
        return "Fix this failing test suite:\n\n" + (tests["stdout"] + tests["stderr"])[-4000:]

    validate = run("PYTHONPATH=src python3 src/validate.py")
    if validate["exit_code"] != 0:
        return "Fix this failing validation check:\n\n" + (validate["stdout"] + validate["stderr"])[-4000:]

    return next_backlog_item()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run -v`
Expected: PASS (24 tests). Note: `test_falls_back_to_backlog_when_checks_pass` takes longer than the others since `fake_run` here doesn't actually skip the real test run inside select_task's OWN test setup — wait, it does: `fake_run` is fully injected, no real subprocess runs in this test. Should be fast.

- [ ] **Step 5: Commit**

```bash
git add scripts/agent_run.py tests/test_agent_run.py
git commit -m "Add select_task(): failing checks first, else next backlog item"
```

---

### Task 6: `agent_run.py` — `verify()`

**Files:**
- Modify: `scripts/agent_run.py`
- Modify: `tests/test_agent_run.py`

**Interfaces:**
- Consumes: `agent_tools.run_command`.
- Produces: `agent_run.verify(run=agent_tools.run_command) -> tuple[bool, str]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_run.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run.TestVerify -v`
Expected: FAIL — `AttributeError: module 'agent_run' has no attribute 'verify'`

- [ ] **Step 3: Add `verify()` to `scripts/agent_run.py`**

```python
# Same three checks, and the same site-table/size thresholds, as
# .github/workflows/ci.yml -- kept in sync by hand since they're a handful
# of lines each; if ci.yml's checks change, update this to match.
_NEEDED_TABLES = [
    "Settle", "Test batting", "ODI batting", "T20I batting",
    "Women's ODI", "Women's T20I", "Women's Test",
    "Test bowling", "ODI bowling", "T20I bowling",
]
_MIN_SITE_BYTES = 300_000


def verify(run=agent_tools.run_command) -> tuple[bool, str]:
    """Re-runs ci.yml's checks: tests, validate.py, then the site build and
    its table-presence check. Stops at the first failure.
    """
    log: list[str] = []

    for name, command in [
        ("tests", "PYTHONPATH=src python3 -m unittest discover -s tests"),
        ("validate", "PYTHONPATH=src python3 src/validate.py"),
        ("site build", "PYTHONPATH=src python3 src/build_site.py"),
    ]:
        out = run(command, timeout=600)
        log.append(f"--- {name} ---\n{out['stdout'][-2000:]}\n{out['stderr'][-2000:]}")
        if out["exit_code"] != 0:
            return False, "\n".join(log)

    html_path = ROOT / "out" / "index.html"
    if not html_path.exists():
        log.append("site build did not produce out/index.html")
        return False, "\n".join(log)

    html = html_path.read_text()
    missing = [t for t in _NEEDED_TABLES if t not in html]
    if missing or len(html) < _MIN_SITE_BYTES:
        log.append(f"site check failed: missing={missing}, len={len(html)}")
        return False, "\n".join(log)

    return True, "\n".join(log)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run -v`
Expected: PASS (29 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/agent_run.py tests/test_agent_run.py
git commit -m "Add verify(): re-run ci.yml's checks before landing any change"
```

---

### Task 7: `agent_run.py` — `run_agent_loop()`

**Files:**
- Modify: `scripts/agent_run.py`
- Modify: `tests/test_agent_run.py`

**Interfaces:**
- Consumes: `agent_tools.TOOL_SCHEMAS`, `agent_tools.dispatch` (Task 3), `verify()` (Task 6).
- Produces: `agent_run.run_agent_loop(task: str, gateway, verify=verify, max_turns=25, max_repair_turns=3) -> tuple[bool, str]`, `agent_run.SYSTEM_PROMPT: str`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_run.py`. This introduces `FakeGateway`, shaped like `src/mock_provider.py`'s `MockProvider` (`.generate()`, `.text()`, `.calls()`), but scripted turn-by-turn rather than by role:

```python
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
        verified, summary = agent_run.run_agent_loop(
            "a task", gw, verify=lambda: next(verify_results), max_repair_turns=3,
        )
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run.TestRunAgentLoop -v`
Expected: FAIL — `AttributeError: module 'agent_run' has no attribute 'run_agent_loop'`

- [ ] **Step 3: Add `SYSTEM_PROMPT` and `run_agent_loop()` to `scripts/agent_run.py`**

```python
import json  # add to the existing import block at the top

MAX_TURNS = 25
MAX_REPAIR_TURNS = 3
STALL_LIMIT = 2

SYSTEM_PROMPT = """You are working autonomously on Howzat, an era-adjusted \
cricket ratings project. You have exactly ONE task this run: the one given \
below. Do only that task, then call finish_task.

Known bug shapes to avoid reintroducing (all were silent -- plausible \
numbers that were wrong, not crashes):
- Fixed column positions instead of mapping columns by header name.
- Name-keyed player identity instead of Statsguru's player_id.
- Era or opposition as free parameters instead of fixed, time-aware offsets.
- Hand-picked shrinkage instead of empirical Bayes.

Validation standard: a rating that disagrees with settled cricket opinion is \
broken, not brave (Bradman first in Tests, exact published career wicket \
totals). Every reported effect needs a 95% confidence interval; never a bare \
number.

Use write_file with the file's COMPLETE new contents, never a diff or a \
partial snippet. Use run_command to run tests as you go. Call finish_task \
exactly once, when the task is done or you are giving up on it -- plain \
text alone never ends the run."""


def run_agent_loop(
    task: str,
    gateway,
    verify=verify,
    max_turns: int = MAX_TURNS,
    max_repair_turns: int = MAX_REPAIR_TURNS,
) -> tuple[bool, str]:
    """Bounded tool-calling loop for one task.

    Returns (verified, summary_or_reason). `verified` False means: no
    landable change came out of this run (stalled, ran out of turns, or
    verify() kept failing after every repair attempt) -- the caller should
    discard any working-tree changes, not commit them.
    """
    history: list[dict] = []
    current = f"Your task:\n\n{task}"
    stalls = 0
    repairs_left = max_repair_turns

    for _ in range(max_turns):
        resp = gateway.generate(
            current, tier="strong", system=SYSTEM_PROMPT,
            tools=agent_tools.TOOL_SCHEMAS, history=history,
        )
        calls = gateway.calls(resp)

        if not calls:
            stalls += 1
            if stalls >= STALL_LIMIT:
                return False, f"stalled: no tool call for {STALL_LIMIT} turns in a row"
            history = history + [{"role": "user", "parts": [{"text": current}]}]
            current = "You must call a tool, including finish_task if you are done. Plain text alone does not end the run."
            continue
        stalls = 0

        results = []
        finish_summary = None
        for call in calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            out = agent_tools.dispatch(name, args)
            results.append(f"{name}({json.dumps(args, sort_keys=True, default=str)}) -> {json.dumps(out, default=str)[:4000]}")
            if name == "finish_task":
                finish_summary = str(args.get("summary", ""))

        if finish_summary is not None:
            passed, log = verify()
            if passed:
                return True, finish_summary
            if repairs_left <= 0:
                return False, f"verification failed after all repair attempts:\n{log[-2000:]}"
            repairs_left -= 1
            history = history + [{"role": "user", "parts": [{"text": current}]}]
            current = f"finish_task was called, but verification failed:\n{log[-3000:]}\n\nFix it, then call finish_task again."
            continue

        history = history + [{"role": "user", "parts": [{"text": current}]}]
        current = "Tool results:\n" + "\n".join(results)

    return False, f"turn budget exhausted ({max_turns} turns) without a verified finish_task"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run -v`
Expected: PASS (34 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/agent_run.py tests/test_agent_run.py
git commit -m "Add run_agent_loop(): bounded tool loop with verify-and-repair"
```

---

### Task 8: `agent_run.py` — `_slugify()` and `land_change()`

**Files:**
- Modify: `scripts/agent_run.py`
- Modify: `tests/test_agent_run.py`

**Interfaces:**
- Produces: `agent_run._slugify(text: str) -> str`, `agent_run.land_change(task: str, summary: str) -> str`.

Note: `land_change()`'s actual `git`/`gh` calls are integration-only — meaningfully mocking `subprocess.run` for every git/gh invocation would test the mock, not the behavior. Only `_slugify()` (the pure part) gets a unit test here; `land_change()` itself is verified by the manual end-to-end run in Task 10.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent_run.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run.TestSlugify -v`
Expected: FAIL — `AttributeError: module 'agent_run' has no attribute '_slugify'`

- [ ] **Step 3: Add `_slugify()` and `land_change()` to `scripts/agent_run.py`**

```python
import subprocess  # add to the existing import block at the top
from datetime import date  # add to the existing import block at the top


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:40].strip("-") or "task"


def land_change(task: str, summary: str) -> str:
    """Commits the working tree, pushes a branch, opens a PR, and enables
    auto-merge on it. Returns the PR URL.

    Assumes the caller already confirmed verify() passed -- this function
    doesn't re-check, it just lands what's already there. Assumes git
    identity is already configured (the workflow does this once, not per
    run) and that the repo has "Allow auto-merge" enabled (see Task 10).
    """
    branch = f"auto/{_slugify(task)}-{date.today().isoformat()}"

    subprocess.run(["git", "checkout", "-b", branch], cwd=ROOT, check=True)
    subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", f"{task}\n\n{summary}"], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "-u", "origin", branch], cwd=ROOT, check=True)

    pr = subprocess.run(
        ["gh", "pr", "create", "--title", task[:70], "--body", summary,
         "--base", "main", "--head", branch],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    pr_url = pr.stdout.strip()

    subprocess.run(
        ["gh", "pr", "merge", "--auto", "--squash", "--delete-branch", pr_url],
        cwd=ROOT, check=True,
    )
    return pr_url
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run -v`
Expected: PASS (37 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/agent_run.py tests/test_agent_run.py
git commit -m "Add land_change(): branch, commit, push, PR, auto-merge"
```

---

### Task 9: `agent_run.py` — `main()`

**Files:**
- Modify: `scripts/agent_run.py`
- Modify: `tests/test_agent_run.py`

**Interfaces:**
- Consumes: everything from Tasks 4-8.
- Produces: `agent_run._write_summary(text: str) -> None`, `agent_run.main() -> int`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent_run.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run.TestWriteSummary -v`
Expected: FAIL — `AttributeError: module 'agent_run' has no attribute '_write_summary'`

- [ ] **Step 3: Add `_write_summary()` and `main()` to `scripts/agent_run.py`**

```python
import os  # add to the existing import block at the top


def _write_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        Path(path).write_text(text)
    print(text)


def main() -> int:
    task = select_task()
    if task is None:
        _write_summary("Nothing queued -- BACKLOG.md's ## Next is empty.")
        return 0

    from gateway import Gateway  # imported here, not at module level, so
    # tests that never construct a real Gateway don't need a working API key.

    gw = Gateway()
    verified, summary = run_agent_loop(task, gw)

    if not verified:
        subprocess.run(["git", "reset", "--hard"], cwd=ROOT)
        subprocess.run(["git", "clean", "-fd"], cwd=ROOT)
        _write_summary(f"No change landed this run.\n\nTask: {task}\n\nWhy: {summary}")
        return 0

    pr_url = land_change(task, summary)
    _write_summary(f"Opened {pr_url}\n\nTask: {task}\n\n{summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agent_run -v`
Expected: PASS (39 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/agent_run.py tests/test_agent_run.py
git commit -m "Add main(): wire task selection, the agent loop, and landing together"
```

---

### Task 10: The workflow, repo settings, and one real end-to-end run

**Files:**
- Create: `.github/workflows/autonomous-build.yml`

- [ ] **Step 1: Enable auto-merge and squash-merge on the repo**

```bash
gh repo edit POLESTARRR/howzat --enable-auto-merge --enable-squash-merge --delete-branch-on-merge
```

- [ ] **Step 2: Confirm the repo secret exists**

The workflow needs `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) available as a repo secret — `src/gateway.py` reads it from the environment, and `.env` isn't present in the Actions runner.

```bash
gh secret list --repo POLESTARRR/howzat
```

If `GOOGLE_API_KEY` (or `GEMINI_API_KEY`/`GOOGLE_AI_STUDIO`) isn't listed, set it from the value already in the local `.env`:

```bash
gh secret set GOOGLE_API_KEY --repo POLESTARRR/howzat --body "$(grep -E '^(GOOGLE_API_KEY|GOOGLE_AI_STUDIO|GEMINI_API_KEY)=' .env | head -1 | cut -d= -f2-)"
```

- [ ] **Step 3: Create `.github/workflows/autonomous-build.yml`**

```yaml
name: Autonomous build

on:
  schedule:
    - cron: "0 */6 * * *"
  workflow_dispatch: {}

permissions:
  contents: write
  pull-requests: write

jobs:
  agent:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install
        run: pip install -q pandas numpy scipy pyarrow beautifulsoup4 lxml requests

      - name: Configure git identity
        run: |
          git config user.name "Dhruv Sharma"
          git config user.email "274071256+POLESTARRR@users.noreply.github.com"

      - name: Run the agent
        env:
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
          GH_TOKEN: ${{ github.token }}
        run: PYTHONPATH=src python3 scripts/agent_run.py
```

- [ ] **Step 4: Verify the YAML parses**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/autonomous-build.yml'))" && echo OK`
Expected: `OK`

(If `pyyaml` isn't installed: `pip install -q pyyaml` first — dev-only, not added to `requirements.txt`/`pyproject.toml` since the workflow itself doesn't need it.)

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/autonomous-build.yml
git commit -m "Add the autonomous-build GitHub Actions workflow"
```

- [ ] **Step 6: Push everything and trigger one real run**

```bash
git push origin main
gh workflow run autonomous-build.yml --repo POLESTARRR/howzat
```

- [ ] **Step 7: Watch it and confirm the outcome**

```bash
gh run watch --repo POLESTARRR/howzat --exit-status
```

Expected: run completes. Check the job summary (`gh run view --repo POLESTARRR/howzat --log` or the Actions tab) for one of:
- A new PR opened at `github.com/POLESTARRR/howzat/pulls`, which should auto-merge once `ci.yml`'s own check on it goes green — watch that separately with `gh pr checks <PR_URL>`.
- "No change landed this run" with a reason — acceptable for a first real run against live BACKLOG.md content and a real (weaker) model; re-read the reason to judge whether it's a one-off or a design gap.
- "Nothing queued" — means BACKLOG.md's `## Next` was already empty when this ran; add an item and re-trigger to actually exercise the path.

If the run fails outright (not "no change landed," an actual workflow failure), read the log — likely causes are the `GOOGLE_API_KEY` secret being missing/wrong, or a `gh` permission error from Step 1/2 not having been applied.

---

## Summary

10 tasks, 4 new files (`scripts/agent_tools.py`, `scripts/agent_run.py`, `tests/test_agent_run.py`, `.github/workflows/autonomous-build.yml`), all building on infrastructure that already exists and already works in this repo (`src/gateway.py`, the `TOOL_SCHEMAS`/`dispatch()` convention, `ci.yml`'s exact checks). No new dependencies, no cost, no Claude anywhere in the path.
