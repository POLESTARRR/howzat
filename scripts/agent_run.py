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
