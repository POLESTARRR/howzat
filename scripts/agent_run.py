"""Orchestrates one autonomous-build run: pick a task, let the model work it
with tools, verify the result, and land it as a PR.

Replaces the "Howzat -- continuous build" Claude Code cloud routine with a
free path that keeps working after the Claude subscription ends: this talks
to Gemini through the project's own src/gateway.py, never Claude.

Run directly: PYTHONPATH=src python3 scripts/agent_run.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import agent_tools  # noqa: E402

_NEXT_ITEM_RE = re.compile(r"^\d+\.\s+(.*)$")

# Same three checks, and the same site-table/size thresholds, as
# .github/workflows/ci.yml -- kept in sync by hand since they're a handful
# of lines each; if ci.yml's checks change, update this to match.
_NEEDED_TABLES = [
    "Settle", "Test batting", "ODI batting", "T20I batting",
    "Women's ODI", "Women's T20I", "Women's Test",
    "Test bowling", "ODI bowling", "T20I bowling",
]
_MIN_SITE_BYTES = 300_000


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
