# Results (auto-generated from declared complete runs)

Runs: 12 complete; missing/failed: 0; duplicates: 0; pairing issues: none.

Five seeds: descriptive only, no significance tests.

## Table 2 — post-change quality and bytes (b=0.25)

| data | scen. | method | post log loss (pooled) | macro | worst clinic | AUROC | adapt. MB | end-to-end MB | client trainings |
|---|---|---|---|---|---|---|---|---|---|
| synthetic | S0 | always_on | 0.4141 | 0.4141 | 0.4319 | 0.8674 | 16.60 | 17.21 | 400 |
| synthetic | S0 | periodic | 0.4195 | 0.4195 | 0.4387 | 0.8642 | 4.15 | 4.76 | 50 |
| synthetic | S0 | random | 0.4172 | 0.4172 | 0.4378 | 0.8654 | 4.15 | 4.76 | 50 |
| synthetic | S0 | drift_only | 0.4341 | 0.4341 | 0.4552 | 0.8624 | 0.25 | 0.86 | 3 |
| synthetic | S0 | known_time | 0.4516 | 0.4516 | 0.4746 | 0.8604 | 0.00 | 0.61 | 0 |
| synthetic | S0 | proposed | 0.4341 | 0.4341 | 0.4552 | 0.8624 | 0.25 | 0.86 | 3 |
| synthetic | S2 | always_on | 0.4868 | 0.4868 | 0.5859 | 0.8242 | 16.60 | 17.21 | 400 |
| synthetic | S2 | periodic | 0.5064 | 0.5064 | 0.6352 | 0.8122 | 4.15 | 4.76 | 50 |
| synthetic | S2 | random | 0.5075 | 0.5075 | 0.6454 | 0.8079 | 4.15 | 4.76 | 50 |
| synthetic | S2 | drift_only | 0.5014 | 0.5014 | 0.6105 | 0.8162 | 1.00 | 1.61 | 12 |
| synthetic | S2 | known_time | 0.5467 | 0.5467 | 0.7291 | 0.8097 | 0.33 | 0.95 | 6 |
| synthetic | S2 | proposed | 0.5014 | 0.5014 | 0.6105 | 0.8162 | 1.00 | 1.61 | 12 |

## Paired differences (proposed − other), post log loss pooled; negative = proposed better

| scen. | comparison | mean diff | sd | seeds proposed better |
|---|---|---|---|---|
| S0 | proposed - periodic | +0.0147 | nan | 0/1 |
| S0 | proposed - drift_only | +0.0000 | nan | 0/1 |
| S0 | proposed - random | +0.0169 | nan | 0/1 |
| S0 | proposed - always_on | +0.0200 | nan | 0/1 |
| S0 | proposed - known_time | -0.0175 | nan | 1/1 |
| S2 | proposed - periodic | -0.0050 | nan | 1/1 |
| S2 | proposed - drift_only | +0.0000 | nan | 0/1 |
| S2 | proposed - random | -0.0061 | nan | 1/1 |
| S2 | proposed - always_on | +0.0146 | nan | 0/1 |
| S2 | proposed - known_time | -0.0453 | nan | 1/1 |

## Drift outcomes (means over seeds)

| dataset   | scenario   | method     |   detected |   n_events |   misses |   mean_detection_delay |   false_alarms_per_100_client_steps |   recovered |   censored |   mean_recovery_time |   expired_requests |   mean_alarm_to_service |   n_client_trainings |
|:----------|:-----------|:-----------|-----------:|-----------:|---------:|-----------------------:|------------------------------------:|------------:|-----------:|---------------------:|-------------------:|------------------------:|---------------------:|
| synthetic | S0         | drift_only |          0 |          0 |        0 |                    nan |                               0.25  |           0 |          0 |                  nan |                  0 |                       0 |                    3 |
| synthetic | S0         | known_time |          0 |          0 |        0 |                    nan |                               0.25  |           0 |          0 |                  nan |                  0 |                     nan |                    0 |
| synthetic | S0         | proposed   |          0 |          0 |        0 |                    nan |                               0.25  |           0 |          0 |                  nan |                  0 |                       0 |                    3 |
| synthetic | S2         | drift_only |          2 |          2 |        0 |                      4 |                               0.543 |           0 |          2 |                  nan |                  0 |                       0 |                   12 |
| synthetic | S2         | known_time |          2 |          2 |        0 |                      4 |                               0.543 |           0 |          2 |                  nan |                  0 |                     nan |                    6 |
| synthetic | S2         | proposed   |          2 |          2 |        0 |                      4 |                               0.543 |           0 |          2 |                  nan |                  0 |                       0 |                   12 |

## Priority ablation

```
                            post_logloss_pooled     post_logloss_worst_client     mean_alarm_to_service     expired_requests     bytes_adaptation    
                                           mean std                      mean std                  mean std             mean std             mean std
dataset   scenario method                                                                                                                            
synthetic S0       proposed              0.4341 NaN                    0.4552 NaN                   0.0 NaN              0.0 NaN         248928.0 NaN
          S2       proposed              0.5014 NaN                    0.6105 NaN                   0.0 NaN              0.0 NaN         995712.0 NaN
```

## Budget sweep

```
                                        post_logloss_pooled  bytes_adaptation  n_client_trainings
dataset   scenario method     budget_b                                                           
synthetic S0       drift_only 0.25                   0.4341          248928.0                 3.0
                   periodic   0.25                   0.4195         4148800.0                50.0
                   proposed   0.25                   0.4341          248928.0                 3.0
          S2       drift_only 0.25                   0.5014          995712.0                12.0
                   periodic   0.25                   0.5064         4148800.0                50.0
                   proposed   0.25                   0.5014          995712.0                12.0
```
