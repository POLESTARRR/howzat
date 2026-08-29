# Standing goal

Set by the repo owner, 2026-08-30. Read this at the start of every run.

## The instruction

Keep working on this project continuously and autonomously. Do not stop to ask
questions and do not wait for a message. Keep going after a session ends, after
a daily limit resets, and while the owner's laptop is asleep. Only stop when the
weekly limit is reached.

The hourly cloud routine exists precisely for this: local loops die with the
session, the cloud routine does not.

## What to work on, in order

1. **Any failing test or validation check.** Nothing else matters until green.
2. **Real bugs.** Hunt actively; see `README.md` for the ones already found and
   the shapes they take. Most were silent — plausible numbers that were wrong.
3. **Grow coverage.** More players, more data, more formats. Missing now: T20I
   bowling ratings, peak ratings for ODI/T20I and for bowling.
4. **Rule-change analysis** (`src/rulechange.py`). What changed, before versus
   after, for every Law and playing-condition change. This is a standing
   priority, not a one-off.
5. Next unfinished item in `BACKLOG.md`.
6. **When the backlog empties, brainstorm.** Promote an idea from `IDEAS.md`,
   or add new ones. An empty queue is not a reason to stop — it is the signal
   to think about what this project could become next. Explore new directions
   actively; do not wait to be told.

## The 95% rule

Every effect this project reports carries a **95% confidence interval**. Never a
bare number. A point estimate without an interval is not a finding, and an
effect is only called significant when its interval excludes zero *and* a
permutation test agrees.

Where two estimators disagree, report both — the disagreement is usually the
real result. Covering pitches in 1972 looks like nothing on a raw before/after
(−0.43, not significant) yet is worth **+2.37 [+0.82, +3.91]** against the
pre-existing trend.

## The standard to hold

A rating that disagrees with settled cricket opinion is broken, not brave.
Bradman first in Tests. Viv Richards first in ODIs. Career wicket totals must
match the published record exactly. If a change breaks those, the change is
wrong — revert it.
