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
