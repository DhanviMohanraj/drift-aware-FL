# Regime study (semisynthetic, configs/v2.yaml)

Periodic interval chosen per budget on dev seeds [101]: {0.05: 5, 0.1: 2}. Final seeds [66, 77, 88, 99, 111].
Post-change pooled log loss, mean of 5 seeds (lower is better). 'better' = seeds where the method beats periodic on the same seed.

## Budget b = 0.05

| Scenario | always-on* | periodic | drift-only | proposed v2 | proposed v3 | v2 better than periodic | v3 better than periodic | v2 MB vs periodic MB |
|---|---|---|---|---|---|---|---|---|
| SS2 | 0.2739 | 0.3091 | 0.3161 | **0.3087** | 0.3153 | 3/5 | 1/5 | 4.10 vs 4.10 |
| SS3 | 0.2739 | 0.3128 | 0.3188 | **0.3112** | 0.3183 | 3/5 | 2/5 | 4.10 vs 4.10 |
| SS6 | 0.2690 | 0.2948 | 0.2995 | **0.2933** | 0.2990 | 4/5 | 2/5 | 4.10 vs 4.10 |

## Budget b = 0.1

| Scenario | always-on* | periodic | drift-only | proposed v2 | proposed v3 | v2 better than periodic | v3 better than periodic | v2 MB vs periodic MB |
|---|---|---|---|---|---|---|---|---|
| SS2 | 0.2739 | 0.3004 | 0.3032 | **0.2947** | 0.2958 | 3/5 | 4/5 | 6.97 vs 8.19 |
| SS3 | 0.2739 | 0.3018 | 0.2975 | **0.2915** | 0.2949 | 5/5 | 5/5 | 7.05 vs 8.19 |
| SS6 | 0.2690 | 0.2898 | 0.2914 | **0.2828** | 0.2846 | 3/5 | 5/5 | 6.97 vs 8.19 |

## Worst-clinic post-change loss (mean of seeds; lower is better)

| Budget | Scenario | periodic | drift-only | proposed v2 | proposed v3 |
|---|---|---|---|---|---|
| 0.05 | SS2 | 0.3472 | 0.3565 | **0.3469** | 0.3568 |
| 0.05 | SS3 | 0.3466 | 0.3525 | **0.3426** | 0.3535 |
| 0.05 | SS6 | 0.3480 | 0.3539 | **0.3440** | 0.3552 |
| 0.1 | SS2 | 0.3348 | 0.3398 | **0.3279** | 0.3287 |
| 0.1 | SS3 | 0.3309 | 0.3262 | **0.3184** | 0.3218 |
| 0.1 | SS6 | 0.3360 | 0.3392 | **0.3245** | 0.3293 |

*always-on is unbudgeted and shown only as a reference.
