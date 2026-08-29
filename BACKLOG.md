# Howzat backlog

Never idle. When bugs are clear, pull the next item. When this empties, add more.

## Done
- [x] CRI+ Test batting (8/8 validation checks)
- [x] CRI+ / SR+ / BAT+ for ODI and T20I
- [x] BOWL+ Test and ODI, with confidence intervals
- [x] Peak-window ratings (`src/peak.py`), with a `peak_distinct` flag
- [x] All-rounder rating (`src/allrounder.py`)
- [x] Debate eval set (`src/evalset.py`)
- [x] Static explorer, four tabs, no backend

## Next
1. [DONE] BOWL+ all three formats.
2. Home/away split as a model term — unmodelled and known to matter.
3. Batting-position covariate: tail-enders drag the era baseline.
4. Expand `src/evalset.py`; run it live and record a baseline score.
5. External validation against published era-adjusted lists.
6. Peak ratings for ODI/T20I and for bowling.

## Known limitations (honest, not bugs)
- CRI+ measures dominance over contemporaries, not absolute skill.
- Test SR+ is unreliable pre-1990s: balls-faced is only ~40% recorded.
- The Full Member filter excludes associate cricket by design.
- Limited-overs weights (0.55/0.45, 0.30/0.70) are a judgement call.
