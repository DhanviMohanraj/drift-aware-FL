# Regime study (clinical, configs/v2.yaml)

Periodic interval chosen per budget on dev seeds [101, 102]: {0.05: 2, 0.1: 2}. Final seeds [11, 22, 33, 44, 55].
Post-change pooled log loss, mean of 5 seeds (lower is better). 'better' = seeds where the method beats periodic on the same seed.

## Budget b = 0.05

| Scenario | always-on* | periodic | drift-only | proposed | proposed v2 | proposed better than periodic | v2 better than periodic | proposed MB vs periodic MB |
|---|---|---|---|---|---|---|---|---|
| C1 | 0.2782 | 0.3535 | 0.3787 | 0.3963 | **0.3501** | 2/5 | 4/5 | 4.01 vs 4.12 |
| C2 | 0.2786 | **0.3540** | 0.3849 | 0.3822 | 0.3604 | 3/5 | 3/5 | 4.09 vs 4.12 |

## Budget b = 0.1

| Scenario | always-on* | periodic | drift-only | proposed | proposed v2 | proposed better than periodic | v2 better than periodic | proposed MB vs periodic MB |
|---|---|---|---|---|---|---|---|---|
| C1 | 0.2782 | 0.3359 | 0.3367 | 0.3360 | **0.3159** | 3/5 | 4/5 | 7.06 vs 8.24 |
| C2 | 0.2786 | 0.3364 | 0.3370 | 0.3394 | **0.3343** | 4/5 | 4/5 | 7.22 vs 8.24 |

*always-on is unbudgeted and shown only as a reference.
