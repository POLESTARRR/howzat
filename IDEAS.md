# Ideas — where this project can still go

Kept deliberately longer than the backlog. When BACKLOG.md empties, promote from
here. When this empties, generate more: **an empty queue is not a reason to stop,
it is the signal to think.**

## Big swings (each is a project in itself)

1. **Cricket WAR — wins above replacement.** The natural successor to CRI+.
   Baseball went wRC+ -> WAR; cricket has no equivalent at all. Convert runs and
   wickets into *wins* over a replacement-level player, so a batter and a bowler
   can finally be compared on one axis. Needs a runs-to-wins conversion
   estimated per format and era.
2. **Women's cricket.** Statsguru covers women's Tests, WODI and WT20I. Almost
   no era-adjusted analysis exists anywhere. Documented research gap: methods
   used for men's cricket "have not yet been investigated in the context of
   women's cricket". Large, unclaimed, and the machinery already exists.
3. **Pressure / leverage.** 40 at 20-3 is worth more than 40 at 300-2. Build a
   leverage index from match state and weight contributions by it.
4. **Fielding and wicketkeeping.** Entirely unmodelled here and largely
   unmodelled anywhere. Catches, stumpings, dismissals per opportunity.

## Model depth

5. Home/away as a model term — known to matter, currently ignored.
6. Batting-position covariate — tail-enders drag the era baseline.
7. Venue effects (park factors): which grounds inflate scoring, with intervals.
8. Aging curves: when do batters peak, and does it differ by format?
9. Head-to-head matchup ratings (batter vs bowler).
10. Team strength by era — an Elo over 150 years.
11. Longevity-adjusted greatness: peak and durability in one number.

## Coverage

12. Domestic first-class: Ranji, County, Sheffield Shield.
13. Franchise T20: IPL, BBL, PSL, The Hundred.
14. Associate cricket as a *separate* rated pool, rather than filtered out.

## Product

15. Side-by-side player comparison in the site.
16. A small read-only API over the ratings.
17. Write-up of method and findings, with limitations stated.
18. Live debate baseline: run `src/evalset.py --live`, record the score, track it.

## Analysis to publish

19. Every remaining rule change, with 95% intervals (standing priority).
20. Which era was hardest to bat in, defended properly rather than asserted.
21. Do era-adjusted ratings agree with published all-time lists? Where not, why?
