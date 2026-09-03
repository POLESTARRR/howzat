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
