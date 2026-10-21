# Results summary — synthetic tiers (21 Oct 2026)

220/220 runs complete (full core S0–S5 × 6 methods × 5 seeds, S3 ablation, S3 budget sweep), 0 failures,
pairing checks clean. Five seeds → descriptive only. Clinical (C0–C2): **not yet measured** (run in Colab).
Full tables: `results/synthetic/tables/results.md`; figures: `results/synthetic/figures/`.

## Main comparison at b = 0.25 (post-change pooled log loss, lower is better; mean over 5 seeds)

| Scenario | always-on (unbudgeted) | periodic | random | drift-only | known-time | proposed | proposed adapt. MB vs periodic |
|---|---|---|---|---|---|---|---|
| S0 no drift | 0.412 | 0.417 | 0.417 | 0.430 | 0.441 | 0.430 | 0.58 vs 4.15 |
| S1 covariate | 0.400 | 0.405 | 0.407 | 0.424 | 0.419 | 0.424 | 0.67 vs 4.15 |
| S2 concept (2 clinics) | 0.470 | 0.486 | 0.488 | 0.497 | 0.527 | 0.497 | 1.21 vs 4.15 |
| S3 staggered concept | 0.463 | 0.481 | 0.484 | 0.484 | 0.503 | 0.484 | 1.63 vs 4.15 |
| S4 gradual concept | 0.437 | 0.468 | 0.469 | 0.484 | 0.593 | 0.482 | 2.00 vs 4.15 |
| S5 prior shift | 0.475 | 0.488 | 0.491 | 0.501 | 0.519 | 0.501 | 1.65 vs 4.15 |

## What the evidence says

1. **Proposed does not beat periodic on quality.** Paired difference (proposed − periodic) is +0.003 to +0.019
   in every scenario; proposed is better on at most 2/5 seeds anywhere. Research question 1: not supported.
2. **It does use far less communication.** Drift-triggered methods spend ~31% of the allowance (0.6–2.0 MB vs
   4.15 MB for periodic/random, which exhaust it). Because quality is not comparable, the plan's bar for a
   "saves communication" claim is not met.
3. **Severity + waiting-time priority has no measurable effect.** Proposed equals drift-only exactly in S0–S3 and
   differs by ≤ 0.0012 in S4/S5; both ablations are identical to proposed on S3. Alarms rarely overlap, so the
   queue almost never holds >1 clinic and alarm-to-service delay is ~0 steps. Research question 2: untestable
   in this regime — the budget is not binding.
4. **Worst-clinic loss is slightly better than periodic** under concept/prior shift (S2 0.573 vs 0.583,
   S3 0.560 vs 0.577, S5 0.542 vs 0.558), but within one seed-SD — descriptive only.
5. **Detector:** 100% detection for concept shifts (S2/S3, delay 2.6–3.8 steps), 95% for gradual (delay 13),
   but only **20% for covariate shift S1** — the frozen loss barely rises when the label conditional is unchanged.
   False alarms 0.6–0.8 per 100 client-steps (target ≤ 1 met).
6. **Budget sweep (S3):** periodic improves with budget (0.511 → 0.481 → 0.476 at b = 0.10/0.25/0.50); drift
   methods are flat above b = 0.25 because three credits per alarm cap their spending.

## Likely cause and options (decide on DEV streams only, then re-freeze)
The 3-credit rule limits adaptation to ~3 local updates per event, so the budget is never contested. To make
the scheduling question testable: more credits per alarm, a tighter budget (b ≤ 0.10), or more simultaneous
events. Otherwise report this honestly as a negative / benchmark result (plan section 1).
