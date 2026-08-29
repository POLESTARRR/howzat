# Howzat

**Cricket's missing era-adjusted batting rating, and a multi-agent debate engine built on top of it.**

Baseball solved cross-era comparison decades ago. `wRC+` sets 100 = league average
and adjusts for park and run environment, so a 130 means the same thing in 2026 as
it did in 1996. Cricket still uses **batting average** — invented in the 1800s and
adjusted for nothing: not era, not opposition, not conditions, not format.

That is why "Bradman vs Tendulkar vs Kohli" is unresolvable. The most-argued
question in the sport has no measuring instrument. This builds one.

## Layers

| Layer | What it is | Status |
|---|---|---|
| **1. CRI+** | Era-adjusted Test batting rating. 100 = average batter of the same era | **built, 8/8 checks** |
| **2. Debate** | Four agents (Statistician, Historian, Skeptic, Judge) argue a question, grounded in Layer 1 | **built + tested offline**; needs a key to run live |
| **3. Verdict card** | Shareable card showing verdict, panel reasoning and full evidence trail | **built** |
| **4. Explorer** | Self-contained static site, no backend | **built** |

## Try it

```bash
./howzat.py top --n 10                  # leaderboard
./howzat.py player Sachin               # one batter
./howzat.py ask "Bradman or Sachin?"    # run a debate (mock if no API key)
./howzat.py site                        # build out/index.html
./howzat.py check                       # 22 tests + 8 validation checks
```

## Layer 1: how CRI+ works

Raw batting average breaks for two reasons.

**Era drift.** Scoring conditions changed — uncovered pitches, bat technology,
ground dimensions, bowling standards. 50 in 1955 is not 50 in 2005.

**Not-outs.** Batting average treats a not-out as though the innings simply
ended. It is really a *right-censored* observation: we know the batter scored at
least R, not exactly R. Averaging censored and uncensored data together is
statistically wrong, and it inflates anyone who bats with the tail.

### The model

Test innings scores are close to geometrically distributed — roughly constant
hazard of dismissal per ball. So model innings runs as Geometric with mean `mu`:

```
log mu_i = intercept + skill[player_i] + era[decade_i] + opp[opposition_i]
```

- Dismissed innings contribute the pmf, `P(X = r)`
- Not-outs contribute the survival function, `P(X > r)` — proper censoring
- Player skills carry a Gaussian prior (ridge), which is **partial pooling**: a
  batter with 8 innings is shrunk toward the mean, one with 200 is not
- Opposition effects are mean-centred for identifiability

```
CRI+ = 100 * exp(skill_player)
```

Fitted by penalised MLE with **analytic gradients** (L-BFGS-B). This is not
optional: with ~1,200 player parameters a numerical gradient needs 1,200 extra
likelihood evaluations per step. Gradients are finite-difference checked to
`<2e-07` relative error in both era modes. The whole fit takes ~2 seconds.

### The finding that shaped the design

The era term was **originally a free parameter**, and it did not work. Fitted
that way, era effects came out *anti-correlated* with observed scoring
(`r = -0.21`): the 1880s averaged 17.5 runs per dismissal — brutally hard — and
the model called them easy.

The cause is confounding. Only 12.7% of qualified players span 3+ decades, so
era and player skill are nearly collinear: low 1880s scoring can be explained as
"a hard era" or as "weak batters" and the likelihood cannot separate them.

The fix is what `wRC+` actually does: **league average is data, not a fitted
parameter.** `era_offsets()` fixes the era term at the observed log scoring
level per decade. Era effects then track reality (1870s `-0.52`, rising to 1940s
`+0.22`), and CRI+ stops being a repaint of raw average (`r` falls 0.993 → 0.975).

The honest consequence, and it is the same claim baseball's `+` metrics make:
**CRI+ measures dominance over contemporaries, not absolute skill across eras.**
That is the strongest claim this data can support.

### Two further corrections

**Shrinkage is estimated, not chosen.** A hand-picked ridge let 31-innings
careers into the all-time top ten. `fit_eb()` iterates to the empirical-Bayes
value (λ = 0.35 hand-picked → **1.904** estimated), since ridge λ corresponds to
a Gaussian prior with λ = 1/(2σ²).

**Ratings ship with uncertainty.** Standard errors come from the diagonal of the
penalised observed information. Conveniently, dismissed and censored innings
share the same second derivative, `mu(1+r)/(1+mu)²`. This is what makes small
samples visibly uncertain rather than silently wrong:

| Player | Innings | CRI+ | 95% interval |
|---|---|---|---|
| SR Tendulkar | 329 | 218 | [194, 244] |
| AC Voges | 31 | 221 | [155, 316] |

Same point estimate, completely different confidence.

## Beyond Test batting

**BOWL+** (`src/bowl_plus.py`) does the same job for bowlers, and decomposes
rather than flattening. Two bowlers with identical averages can differ entirely
in *how* they are good, so wicket-taking and economy are modelled separately,
per ball, each era-adjusted:

```
wickets_i ~ Poisson(balls_i * exp(strike[bowler] + era + opp))
runs_i    ~ Poisson(balls_i * exp(econ[bowler]   + era + opp))
```

Bowling average is exactly the ratio of those two rates, so overall skill is
`strike − econ` in log space and the trade-off falls out of the arithmetic
rather than being imposed. `BOWL+ = 100 * exp(strike − econ)`, higher is better,
same 100-is-average scale as CRI+.

**Limited overs** (`src/formats.py`). Tests are a longevity format, so runs per
dismissal is nearly the whole story. Limited overs is not: 30 off 15 can beat
40 off 40, and a durability-only rating would rank blockers above match-winners.
So ODI and T20I carry two era-relative indices — `CRI+` (durability) and `SR+`
(tempo) — combined by a weighted geometric mean:

| Format | Durability | Tempo |
|---|---|---|
| Test | 1.00 | 0.00 |
| ODI | 0.55 | 0.45 |
| T20I | 0.30 | 0.70 |

**Those weights are a judgement call, not a derived result**, which is why they
are stated here rather than buried, and why CRI+ and SR+ always ship alongside
the combined number for anyone who disagrees.

### The associate-nation problem

Ratings only count innings against **ICC Full Members**, and that filter is
load-bearing. Since the ICC granted T20I status to every member in 2019, only
**34% of T20I innings (22,768 of 66,624)** are against Full Members; the rest
are associate sides playing each other.

Unfiltered, the all-time T20I table was topped by batters from **Austria,
Bulgaria, Spain and Saudi Arabia** — fast scorers against weak attacks. No
opposition adjustment repairs this, because associate teams mostly play *each
other*, so the baseline sinks with them. Two further guards:

- SR+ is adjusted for **opposition**, not just era. CRI+ already was; without
  the same treatment the combined rating inherited the bias.
- SR+ is **shrunk toward 100** by balls faced (prior weight 600 balls), so a
  22-innings career cannot top the tempo index.

After the fix the T20I top order reads Abhishek Sharma, Kohli, Suryakumar Yadav,
Salt, Buttler, Maxwell, Brook — which is a recognisable list.

## Data

| Source | Coverage | Why |
|---|---|---|
| ESPNcricinfo Statsguru | every Test/ODI/T20I innings, batting **and** bowling | the era-adjustment substrate |
| Cricsheet | 916 Tests, ball-by-ball, 2001–2026 | modern detail |

**Cricsheet cannot do this job alone** — it has *zero* pre-2000 Tests, so it
cannot see the Bradman era at all. That is the single fact that dictates the
architecture.

Raw HTML is cached under `data/raw/statsguru/` (531 MB, gitignored), so re-runs
never refetch. The scraper retries with backoff and a bounded socket timeout;
an unbounded one silently hung a 40-minute run.

## Layer 2: the debate

Four agents argue the question, each with tool access to CRI+:

| Agent | Job | Tier |
|---|---|---|
| **Statistician** | Gather decisive evidence via tool calls | cheap |
| **Historian** | Era context — what the adjustment is correcting for | cheap |
| **Skeptic** | **Attack** the emerging consensus | cheap |
| **Judge** | Weigh it, deliver verdict + confidence | strong |

The Skeptic is not decoration. The documented failure mode of multi-agent debate
is confident agreement on a wrong answer, so one agent is instructed to break the
consensus rather than help build it.

**Grounding.** Agents may only assert numbers a tool returned.
`Transcript.ungrounded_numbers()` reports any figure in the verdict that no tool
produced, and it is printed on the verdict card **even when unflattering**. The
check tolerates honest rounding (335.9 → "336") by matching numerically, but
catches invention.

**Abstention.** `compare()` reports `intervals_separated`. Where two players'
95% intervals overlap — Tendulkar vs Kallis, for instance — the tool explicitly
instructs the panel not to declare a winner. The system is built to say *the data
cannot settle this*.

**Gateway.** Model tiering (only the Judge escalates), on-disk response caching,
retry with backoff, and per-call cost accounting. A full four-agent debate costs
about **$0.003**.

Two things the gateway learned the hard way:

- Google's `/models` endpoint **lists models that 404 on use** ("no longer
  available to new users"), and Pro tiers can be quota-blocked (429) on a key
  where Flash works fine. So tiers are **fallback chains, probed live and
  cached** — never a hardcoded model name. On a persistent failure `generate`
  falls through the rest of the chain rather than aborting a debate that has
  already paid for its evidence.
- Keys are resolved by **trying each candidate until one authenticates**, with
  `.env` checked *before* the environment. A stale `GOOGLE_API_KEY` exported in
  a shell will otherwise shadow the good key in `.env`, and the only symptom is
  a 401 that looks like the new key is broken.

API keys are redacted from every error, log and traceback.

### A bug the system found in itself

The debate was asked *"Was Viv Richards a better ODI batsman than Kohli?"* and
the Skeptic objected: **"everyone is relying on Test ratings to settle an ODI
debate."** It was right. ODI and T20I ratings existed but were never wired into
the tools, so every lookup silently returned Test numbers. Tools now take an
explicit `format`, and the agents are instructed to pass it on every call.

That is the argument for building the Skeptic as an adversary rather than a
reviewer: an agreeable panel would have shipped the wrong answer confidently.

**Testing without a key.** `MockProvider` speaks the gateway's exact response
shape, so the whole loop — tool dispatch, evidence accumulation, role sequencing,
JSON parsing, grounding — is tested offline. It deliberately injects one invented
number so the grounding guard is proven to fire. A guard that has never fired is
not a guard.

## Layout

```
howzat.py                CLI: rate / player / top / ask / site / check
src/fetch_statsguru.py   innings scraper (Test/ODI/T20I), cached + retrying
src/cri_plus.py          the censored geometric model
src/validate.py          8 sanity checks against settled cricket opinion
src/tools.py             data tools the agents call, + JSON schemas
src/gateway.py           LLM gateway: tiering, caching, cost accounting
src/debate.py            four-agent orchestration + grounding
src/mock_provider.py     offline stand-in so Layer 2 is testable without a key
src/verdict_card.py      shareable HTML verdict card
src/build_site.py        self-contained static explorer
tests/test_howzat.py     22 tests
```

## Validation

A rating that disagrees with settled opinion is not brave, it is broken. The
checks are deliberately about cases nobody argues over, so the metric can be
trusted on the cases people *do* argue over. **8/8 currently pass.**

| # | Check | Result |
|---|---|---|
| 1 | Bradman is #1, by a clear margin | 336, **1.37×** clear of Hobbs |
| 2 | Era effects directionally sane | 1870s −0.52 → 1940s +0.22 |
| 3 | Canonical greats rank highly | **20/20** in the top 100 |
| 4 | Shrinkage bites | sd(<30 inns) 38.4 < sd(100+) 49.7 |
| 5 | Disagrees with raw average | r = 0.975 (in `0.6–0.985`) |
| 6 | Uncertainty scales with sample | CI width 74 > 61 > 53 > 49 |
| 7 | Bradman survives uncertainty | lower bound clears **100%** of the field |
| 8 | Era corrections are explicable | 19th-c. batters rise, 1930s–40s fall |

Check 5 matters most: CRI+ must *disagree* with raw average somewhere, or it
adds nothing. Check 7 is the headline — even at the pessimistic end of his
interval, Bradman beats every other batter's point estimate.

### Top 10

| # | Player | Inns | Avg | CRI+ | 95% CI |
|---|---|---|---|---|---|
| 1 | DG Bradman | 80 | 99.9 | **336** | [269, 420] |
| 2 | JB Hobbs | 102 | 56.9 | 246 | [202, 299] |
| 3 | RG Pollock | 41 | 61.0 | 244 | [181, 328] |
| 4 | GS Sobers | 160 | 57.8 | 241 | [205, 284] |
| 5 | KF Barrington | 131 | 58.7 | 240 | [201, 287] |
| 6 | Hon. FS Jackson | 33 | 48.8 | 240 | [172, 333] |
| 7 | ED Weekes | 81 | 58.6 | 238 | [192, 296] |
| 8 | CL Walcott | 74 | 56.7 | 238 | [189, 299] |
| 9 | SPD Smith | 223 | 56.0 | 229 | [200, 264] |
| 10 | H Sutcliffe | 84 | 60.7 | 229 | [184, 284] |

Biggest era corrections: **AC Bannerman** rises 332 rank places (averaged 23.1 in
the 1880s, when 23 was a strong return); Zimbabwe-era batters fall, because the
opposition term discounts runs made against weak attacks.

## Running

```bash
python3 src/fetch_statsguru.py 1877 2026   # ~25 min cold, instant cached
python3 src/validate.py                     # fits the model, prints the report
```

## Layer 2 status

`src/gateway.py` implements model tiering (cheap tier for evidence-gathering
agents, strong tier for the judge), on-disk request caching, retry with backoff,
and per-call cost accounting so a debate can report **cost per verdict**.

It is currently **blocked**: the `GOOGLE_API_KEY` in the environment returns
`401 Unauthorized`. It is an `AQ.`-prefixed OAuth-style token, not an `AIza`
Gemini API key. A valid key from https://aistudio.google.com/apikey unblocks it.
