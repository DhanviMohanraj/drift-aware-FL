# Regime study (clinical, configs/v2.yaml)

Periodic interval chosen per budget on dev seeds [101, 102]: {0.05: 2, 0.1: 2}. Final seeds [11, 22, 33, 44, 55].
Post-change pooled log loss, mean of 5 seeds (lower is better). 'better' = seeds where the method beats periodic on the same seed.

## Budget b = 0.05

| Scenario | always-on* | periodic | drift-only | proposed | proposed v2 | proposed v3 | v2 better than periodic | v3 better than periodic | v3 MB vs periodic MB |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 0.2782 | 0.3535 | 0.3787 | 0.3963 | **0.3501** | 0.3532 | 4/5 | 3/5 | 4.12 vs 4.12 |
| C2 | 0.2786 | **0.3540** | 0.3849 | 0.3822 | 0.3604 | 0.3590 | 3/5 | 3/5 | 4.12 vs 4.12 |

## Budget b = 0.1

| Scenario | always-on* | periodic | drift-only | proposed | proposed v2 | proposed v3 | v2 better than periodic | v3 better than periodic | v3 MB vs periodic MB |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 0.2782 | 0.3359 | 0.3367 | 0.3360 | **0.3159** | 0.3219 | 4/5 | 5/5 | 6.89 vs 8.24 |
| C2 | 0.2786 | 0.3364 | 0.3370 | 0.3394 | 0.3343 | **0.3241** | 4/5 | 3/5 | 7.22 vs 8.24 |

## Worst-clinic post-change loss (mean of seeds; lower is better)

| Budget | Scenario | periodic | drift-only | proposed | proposed v2 | proposed v3 |
|---|---|---|---|---|---|---|
| 0.05 | C1 | **0.6294** | 0.7209 | 0.7675 | 0.6407 | 0.6325 |
| 0.05 | C2 | **0.6170** | 0.7512 | 0.7450 | 0.6867 | 0.6395 |
| 0.1 | C1 | **0.5940** | 0.6687 | 0.6679 | 0.6091 | 0.5940 |
| 0.1 | C2 | **0.5803** | 0.6722 | 0.6804 | 0.6704 | 0.5831 |

*always-on is unbudgeted and shown only as a reference.
