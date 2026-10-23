# Regime study (clinical, configs/v2.yaml)

Periodic interval chosen per budget on dev seeds [101, 102]: {0.05: 2, 0.1: 2}. Final seeds [66, 77, 88, 99, 111].
Post-change pooled log loss, mean of 5 seeds (lower is better). 'better' = seeds where the method beats periodic on the same seed.

## Budget b = 0.05

| Scenario | always-on* | periodic | drift-only | proposed | proposed v2 | proposed v3 | v2 better than periodic | v3 better than periodic | v2 MB vs periodic MB |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 0.2730 | 0.3416 | 0.3354 | **0.3330** | 0.3345 | 0.3357 | 3/5 | 4/5 | 4.10 vs 4.10 |
| C2 | 0.2740 | 0.3424 | 0.3275 | 0.3307 | **0.3266** | 0.3319 | 2/5 | 4/5 | 4.10 vs 4.10 |

## Budget b = 0.1

| Scenario | always-on* | periodic | drift-only | proposed | proposed v2 | proposed v3 | v2 better than periodic | v3 better than periodic | v2 MB vs periodic MB |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 0.2730 | **0.3062** | 0.3148 | 0.3139 | 0.3133 | 0.3082 | 2/5 | 2/5 | 7.86 vs 8.19 |
| C2 | 0.2740 | 0.3106 | 0.3155 | 0.3151 | 0.3128 | **0.3063** | 2/5 | 3/5 | 7.78 vs 8.19 |

## Worst-clinic post-change loss (mean of seeds; lower is better)

| Budget | Scenario | periodic | drift-only | proposed | proposed v2 | proposed v3 |
|---|---|---|---|---|---|---|
| 0.05 | C1 | 0.5969 | 0.5774 | **0.5596** | 0.5833 | 0.5855 |
| 0.05 | C2 | 0.6137 | **0.5673** | 0.5901 | 0.5699 | 0.5819 |
| 0.1 | C1 | **0.5239** | 0.5477 | 0.5394 | 0.5448 | 0.5359 |
| 0.1 | C2 | 0.5331 | 0.5525 | 0.5544 | 0.5471 | **0.5219** |

*always-on is unbudgeted and shown only as a reference.
