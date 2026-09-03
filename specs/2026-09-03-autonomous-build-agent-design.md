# Self-hosted replacement for the Claude Code "continuous build" routine

## Why

"Howzat — continuous build" is a Claude Code cloud routine that fires hourly,
reads GOAL.md/BACKLOG.md, and autonomously fixes bugs / adds features / grows
coverage, then commits and pushes. It is not an app calling Claude as a
backend — it *is* a Claude Code cloud session, so it stops firing entirely the
moment the Claude subscription ends, with no config flag to redirect it
elsewhere.

This spec designs a free, self-hosted replacement so the project keeps
getting autonomous work done without a subscription. It reuses
`src/gateway.py`, the project's existing Gemini client (retries, 429 backoff,
model-tier fallback, and native function/tool-calling already wired up for
the debate feature), rather than introducing a new LLM dependency.

## Scope decisions (already made, with the owner)

- **Runs on GitHub Actions**, not locally — free tier, scheduled cron, and a
  built-in `GITHUB_TOKEN` with write access, so the exact "Claude doesn't
  have GitHub access" 403 that broke the original routine can't recur here.
- **Deterministic task selection**, not open-ended bug-hunting. A weaker
  free-tier model is far less reliable at the kind of judgment call GOAL.md's
  own "silent, plausible-looking wrong numbers" warning describes. Each run
  does exactly one of:
  1. If `unittest discover` or `validate.py` is failing, fix that. Nothing
     else runs this cycle.
  2. Otherwise, the first unchecked item under BACKLOG.md's `## Next`
     heading.

  Freeform bug-hunting and promoting ideas from IDEAS.md stay human-triggered
  — not something this agent does on its own.
- **Lands changes via PR + auto-merge, not a direct push to main.** The
  agent's own in-run check pass is not treated as sufficient; the repo's real
  `ci.yml`, running independently on the PR, is the actual gate before
  merge.
- **Commits authored as the repo owner**, matching every existing commit
  (`Dhruv Sharma <274071256+POLESTARRR@users.noreply.github.com>`), no bot
  identity, no AI-attribution trailer — consistent with the existing
  convention in this repo and the owner's standing rule against AI
  attribution on their own work.

## Architecture

```
.github/workflows/autonomous-build.yml   (cron: every 6 hours)
  -> checkout, setup-python (matches ci.yml)
  -> pip install (matches ci.yml's deps)
  -> python3 scripts/agent_run.py
       1. select_task()            -> str | None
       2. run_agent_loop(task)     -> bool (made a change?)
       3. verify()                 -> bool (tests + validate.py + site check)
       4. if verified: open branch, commit, push, open PR, enable auto-merge
          else: git reset --hard, log why, exit 0 (no PR this run)
```

### `select_task()`

1. Run `PYTHONPATH=src python3 -m unittest discover -s tests` and
   `PYTHONPATH=src python3 src/validate.py`. If either exits non-zero,
   the task is "fix this failure," with the captured stderr/stdout attached
   verbatim.
2. Else, parse BACKLOG.md's `## Next` section: numbered/bulleted lines,
   skip any already marked `[DONE]`, return the text of the first
   unmarked one.
3. Else (backlog fully checked off): return `None`. The workflow exits
   cleanly with a job-summary note ("nothing queued") and does not call
   the model at all — no wasted API quota on an empty queue.

### `run_agent_loop(task)`

A bounded tool-calling loop against `Gateway.generate(tier="strong", ...)`
(`gemini-3.7-flash` — the accuracy bar this project holds itself to,
e.g. exact wicket-total matches, justifies the stronger tier over "cheap").

**Tools exposed to the model:**
- `read_file(path)` — returns file content. Path must resolve inside the
  repo root; anything else is a tool error, not a crash.
- `write_file(path, content)` — whole-file overwrite. Chosen over
  patch/diff application: a smaller model producing a malformed diff is a
  much likelier failure mode than a bad whole-file write, and whole-file
  writes are trivial to make idempotent and safe to retry.
- `run_command(command)` — shell, cwd pinned to repo root, 300s timeout.
  Unrestricted beyond that: this executes inside a disposable Actions
  container, not the owner's machine, so the sandboxing that matters is
  "throwaway VM," not command allow-listing.
- `finish_task(summary)` — the model must call this explicitly to end the
  loop. Absence of a further tool call is never treated as "done," to
  avoid guessing intent from silence.

**Turn budget: 25 turns hard cap.** Each model response that returns one or
more `functionCall`s executes them locally and appends the result(s) to
history as `functionResponse` turns, then loops. A response with no
function call and no `finish_task` is treated as a wasted turn (prompted
once to use a tool or finish); repeating that twice in a row ends the loop
early as "stalled," same handling as hitting the cap.

**Prompt content**, condensed from the current routine's instructions:
kept — the known-bug-shape list (column-position drift, name-keyed
identity collisions, era/opposition as free parameters, etc.), the
validation standard (Bradman first, exact career wicket totals), the 95%
CI + permutation-test rule. Dropped — "hunt for bugs" open-ended framing,
brainstorming from IDEAS.md, anything that asks the model to decide *what*
to work on (that's `select_task()`'s job now, not the model's).

### `verify()`

Re-runs the exact three checks `ci.yml` runs: `unittest discover`,
`validate.py`, and the site-build + table-presence check. If any fail, the
loop gets up to 3 repair turns (the failure output fed back as a new tool
result) before giving up. On final failure: `git reset --hard` to a clean
tree, write a summary to the job log, exit 0. This is not treated as a
workflow failure — an unproductive run is the expected, safe outcome of a
smaller model on a hard task, not an error condition.

### Landing the change

On success: create branch `auto/<task-slug>-<YYYY-MM-DD>`, commit (git
identity set to the owner's, matching existing history; message states
what changed and why, no AI attribution), push, `gh pr create`, then
`gh pr merge --auto --squash --delete-branch`. The actual merge only
happens once `ci.yml`'s own `pull_request`-triggered run passes
independently — a genuine second opinion, not a re-statement of the same
in-job result.

## Testing the harness itself

New `tests/test_agent_run.py`, same `unittest` style as the rest of the
repo:
- BACKLOG.md `## Next` parsing (including the `[DONE]`-skipping logic)
- Turn-budget cutoff and "stalled" detection
- Tool execution: path containment for `read_file`/`write_file`, command
  timeout for `run_command`
- All against a fake `Gateway` returning canned `functionCall` responses —
  no real API calls, no quota spent, runs in CI like everything else.

## Explicitly out of scope

- Freeform bug-hunting / IDEAS.md brainstorming (stays human-triggered)
- Any change to the existing Claude Code routine (separate concern; fixing
  its GitHub App access is tracked separately, not part of this spec)
- Rate-limit tuning beyond the initial 6-hour cadence — revisit if Gemini
  free-tier quota turns out to be the binding constraint in practice
