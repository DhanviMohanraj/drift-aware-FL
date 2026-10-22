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

---

# Clinical results — UCI 296, run in Colab (21 Oct 2026)

130/130 clinical runs complete (C0–C2 × 6 methods × 5 seeds, C1 ablation, C1 budget sweep), 0 failures.
Dev calibration (`results/calibration_clinical.json`): threshold 6 (0.5 false alarms/100 dev client-steps,
target met) and periodic interval 2. Tables: `results/combined/tables/results.md`. (The "3 pairing issues"
printed in Colab were a false alarm: synthetic and clinical legitimately use different configs because of
`dataset_overrides`; the check now compares within each dataset.)

| Scenario | always-on | periodic | random | drift-only | known-time | proposed |
|---|---|---|---|---|---|---|
| C0 no drift | 0.334 | 0.393 | 0.447 | 0.771 | 0.583 | 0.771 |
| C1 shift, clinics 1–2 | 0.334 | 0.394 | 0.449 | 0.741 | 0.535 | 0.741 |
| C2 staggered shift | 0.335 | 0.396 | 0.454 | 0.782 | 0.613 | 0.782 |

Post-change pooled log loss, mean of 5 seeds; pooled AUROC 0.55–0.58 for every method.

## What the clinical evidence says — the experiment is not yet informative

1. **The injected drift has no measurable effect.** Always-on loss is identical in C0, C1 and C2 (0.334–0.335).
   A +1 SD shift in one feature (num_lab_procedures) does not change the task.
2. **So the detector never fires on it:** 0/2 events detected in C1, 0/3 in C2. Every alarm (~1 per run) is a
   false alarm. Dev calibration already showed this (20% dev detection).
3. **Drift-triggered methods therefore almost never retrain** (3 local updates in 100 steps) and stay on the
   warm-start model → much worse than periodic (+0.35 to +0.39 log loss, 0/5 seeds better).
4. **Proposed = drift-only again**, and both ablations equal proposed.
5. **Severe label skew between clinics:** in each seed one clinic gets log loss 2–4 (worst-clinic mean 2.2 even
   for always-on). Dirichlet(α = 0.5) on a ~9%-positive binary target makes one clinic mostly positives, which a
   single global model fits badly. AUROC 0.55–0.58 is close to chance.

## Required protocol decisions before the clinical comparison means anything (dev patients only)
- A clinically meaningful drift: larger or multi-feature measurement shift, or a label-related change.
- Milder clinic heterogeneity (e.g. Dirichlet α ≥ 1, or allocation by a feature) — this changes the plan's
  protocol, so agree it with the supervisor.
- More service credits per alarm (also needed for the synthetic priority question).

---

# v2 synthetic results (configs/v2.yaml, 22 Oct 2026)

220/220 runs complete (full S0–S5 × 6 methods × 5 seeds, S3 ablation, S3 budget sweep), 0 failures.
Only change vs earlier synthetic runs: 6 service credits per alarm (chosen on 4 dev seeds).
Tables: `results/synthetic_v2/tables/results.md`; figures: `results/synthetic_v2/figures/`.

Post-change pooled log loss at b = 0.25 (mean of 5 seeds; lower is better):

| Scenario | periodic | drift-only | proposed | proposed − periodic (seeds proposed better) |
|---|---|---|---|---|
| S0 no drift | **0.417** | 0.432 | 0.432 | +0.015 (0/5) |
| S1 covariate | **0.405** | 0.421 | 0.421 | +0.015 (0/5) |
| S2 concept, 2 clinics | **0.486** | 0.488 | 0.489 | +0.002 (2/5) |
| S3 staggered concept | 0.481 | **0.475** | **0.475** | −0.006 (3/5) |
| S4 gradual | **0.468** | 0.478 | 0.479 | +0.011 (0/5) |
| S5 prior shift | 0.488 | 0.486 | **0.486** | −0.002 (4/5) |

- **Drift-triggered retraining now competes with periodic** and wins under staggered/local drift (S3, S5),
  while spending ~45% fewer adaptation bytes (2.3 MB vs 4.15 MB). It loses when there is no drift, covariate
  drift, or gradual drift.
- **Worst-clinic loss is clearly better** for drift-triggered methods under concept/prior drift
  (S3: 0.536 vs 0.577; S2: 0.542 vs 0.582; S5: 0.527 vs 0.558).
- **S3 budget sweep:** drift-triggered beats periodic at every budget (b = 0.10: 0.491 vs 0.511;
  b = 0.50: 0.473 vs 0.476) and at b = 0.50 uses 3.2 MB vs 8.3 MB.
- **The proposed priority rule still adds nothing measurable:** proposed − drift-only is within ±0.001 everywhere;
  the ablations (no severity 0.4753, no waiting 0.4737) are no worse than the full rule (0.4754).
- Five seeds: descriptive only. Clinical v2: running in Colab.
