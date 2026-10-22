# Regime study (synthetic, configs/v2.yaml)

Periodic interval chosen per budget on dev seeds [101, 102, 103, 104]: {0.05: 2, 0.1: 5, 0.25: 2}. Final seeds [11, 22, 33, 44, 55].
Post-change pooled log loss, mean of 5 seeds (lower is better). 'better' = seeds where the method beats periodic on the same seed.

## Budget b = 0.05

| Scenario | always-on* | periodic | random | drift-only | proposed | known-time | proposed better than periodic | proposed MB vs periodic MB |
|---|---|---|---|---|---|---|---|---|
| S2 | 0.4701 | 0.5357 | 0.5327 | 0.5309 | **0.5214** | 0.5155 | 5/5 | 0.82 vs 0.83 |
| S3 | 0.4628 | 0.5406 | 0.5351 | 0.5297 | **0.5231** | 0.5015 | 4/5 | 0.80 vs 0.83 |
| S5 | 0.4750 | 0.5424 | 0.5495 | **0.5158** | 0.5239 | 0.5464 | 5/5 | 0.79 vs 0.83 |
| S6 | 0.4505 | 0.4862 | 0.4880 | 0.4860 | **0.4800** | 0.4788 | 5/5 | 0.82 vs 0.83 |

## Budget b = 0.1

| Scenario | always-on* | periodic | random | drift-only | proposed | known-time | proposed better than periodic | proposed MB vs periodic MB |
|---|---|---|---|---|---|---|---|---|
| S2 | 0.4701 | 0.5105 | 0.5054 | 0.5000 | **0.4981** | 0.4984 | 4/5 | 1.52 vs 1.66 |
| S3 | 0.4628 | 0.5087 | 0.5034 | **0.4905** | 0.4924 | 0.4780 | 5/5 | 1.56 vs 1.66 |
| S5 | 0.4750 | 0.5091 | 0.5123 | 0.5030 | **0.5019** | 0.5187 | 4/5 | 1.53 vs 1.66 |
| S6 | 0.4505 | 0.4729 | 0.4744 | **0.4696** | 0.4731 | 0.4787 | 3/5 | 1.37 vs 1.66 |

## Budget b = 0.25

| Scenario | always-on* | periodic | random | drift-only | proposed | known-time | proposed better than periodic | proposed MB vs periodic MB |
|---|---|---|---|---|---|---|---|---|
| S2 | 0.4701 | **0.4864** | 0.4881 | 0.4880 | 0.4885 | 0.4964 | 2/5 | 2.28 vs 4.15 |
| S3 | 0.4628 | 0.4813 | 0.4840 | **0.4753** | 0.4754 | 0.4791 | 3/5 | 2.98 vs 4.15 |
| S5 | 0.4750 | 0.4880 | 0.4911 | 0.4862 | **0.4856** | 0.5017 | 4/5 | 3.08 vs 4.15 |
| S6 | 0.4505 | **0.4597** | 0.4626 | 0.4676 | 0.4680 | 0.4840 | 1/5 | 1.76 vs 4.15 |

*always-on is unbudgeted (16.6 MB) and shown only as a reference.
