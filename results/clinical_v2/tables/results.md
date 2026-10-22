# Results (auto-generated from declared complete runs)

Runs: 130 complete; missing/failed: 0; duplicates: 0; pairing issues: none.

Five seeds: descriptive only, no significance tests.

## Table 2 — post-change quality and bytes (b=0.25)

| data | scen. | method | post log loss (pooled) | macro | worst clinic | AUROC | adapt. MB | end-to-end MB | client trainings |
|---|---|---|---|---|---|---|---|---|---|
| clinical | C0 | always_on | 0.2722 ± 0.0101 | 0.2977 ± 0.0239 | 0.5112 ± 0.0887 | 0.6178 ± 0.0090 | 82.40 | 85.42 | 400 |
| clinical | C0 | periodic | 0.2927 ± 0.0151 | 0.3208 ± 0.0318 | 0.5512 ± 0.1031 | 0.5949 ± 0.0124 | 20.60 | 23.63 | 50 |
| clinical | C0 | random | 0.2876 ± 0.0162 | 0.3139 ± 0.0304 | 0.5310 ± 0.1005 | 0.5968 ± 0.0130 | 20.60 | 23.63 | 50 |
| clinical | C0 | drift_only | 0.3381 ± 0.0660 | 0.3734 ± 0.0918 | 0.6618 ± 0.2610 | 0.5918 ± 0.0272 | 4.12 | 7.15 | 10 |
| clinical | C0 | known_time | 0.3418 ± 0.0352 | 0.3778 ± 0.0554 | 0.6645 ± 0.1570 | 0.5967 ± 0.0151 | 0.00 | 3.03 | 0 |
| clinical | C0 | proposed | 0.3381 ± 0.0660 | 0.3734 ± 0.0918 | 0.6618 ± 0.2610 | 0.5918 ± 0.0272 | 4.12 | 7.15 | 10 |
| clinical | C1 | always_on | 0.2790 ± 0.0107 | 0.3068 ± 0.0275 | 0.5271 ± 0.0897 | 0.6096 ± 0.0101 | 82.40 | 85.42 | 400 |
| clinical | C1 | periodic | 0.3039 ± 0.0156 | 0.3332 ± 0.0325 | 0.5585 ± 0.0709 | 0.5858 ± 0.0085 | 20.60 | 23.63 | 50 |
| clinical | C1 | random | 0.2966 ± 0.0172 | 0.3250 ± 0.0328 | 0.5528 ± 0.0989 | 0.5883 ± 0.0188 | 20.60 | 23.63 | 50 |
| clinical | C1 | drift_only | 0.3224 ± 0.0234 | 0.3584 ± 0.0485 | 0.6375 ± 0.1737 | 0.5935 ± 0.0211 | 10.59 | 13.62 | 26 |
| clinical | C1 | known_time | 0.3250 ± 0.0378 | 0.3608 ± 0.0605 | 0.6559 ± 0.2127 | 0.5924 ± 0.0294 | 7.14 | 10.17 | 20 |
| clinical | C1 | proposed | 0.3171 ± 0.0228 | 0.3518 ± 0.0446 | 0.6277 ± 0.1649 | 0.5942 ± 0.0227 | 10.59 | 13.62 | 26 |
| clinical | C1 | proposed_nosev | 0.3224 ± 0.0234 | 0.3584 ± 0.0485 | 0.6375 ± 0.1737 | 0.5935 ± 0.0211 | 10.59 | 13.62 | 26 |
| clinical | C1 | proposed_nowait | 0.3240 ± 0.0417 | 0.3616 ± 0.0681 | 0.6292 ± 0.1828 | 0.5936 ± 0.0276 | 9.76 | 12.79 | 24 |
| clinical | C2 | always_on | 0.2801 ± 0.0124 | 0.3059 ± 0.0290 | 0.5120 ± 0.0871 | 0.6096 ± 0.0150 | 82.40 | 85.42 | 400 |
| clinical | C2 | periodic | 0.3046 ± 0.0182 | 0.3329 ± 0.0356 | 0.5497 ± 0.0893 | 0.5886 ± 0.0128 | 20.60 | 23.63 | 50 |
| clinical | C2 | random | 0.2991 ± 0.0217 | 0.3248 ± 0.0378 | 0.5340 ± 0.1008 | 0.5848 ± 0.0209 | 20.60 | 23.63 | 50 |
| clinical | C2 | drift_only | 0.3164 ± 0.0310 | 0.3488 ± 0.0504 | 0.6259 ± 0.1888 | 0.5799 ± 0.0362 | 12.65 | 15.68 | 31 |
| clinical | C2 | known_time | 0.3535 ± 0.0492 | 0.3963 ± 0.0826 | 0.7443 ± 0.2596 | 0.5773 ± 0.0384 | 12.36 | 15.39 | 30 |
| clinical | C2 | proposed | 0.3167 ± 0.0310 | 0.3492 ± 0.0506 | 0.6264 ± 0.1889 | 0.5786 ± 0.0362 | 12.65 | 15.68 | 31 |

## Paired differences (proposed − other), post log loss pooled; negative = proposed better

| scen. | comparison | mean diff | sd | seeds proposed better |
|---|---|---|---|---|
| C0 | proposed - periodic | +0.0455 | 0.0547 | 1/5 |
| C0 | proposed - drift_only | +0.0000 | 0.0000 | 0/5 |
| C0 | proposed - random | +0.0506 | 0.0521 | 1/5 |
| C0 | proposed - always_on | +0.0660 | 0.0568 | 0/5 |
| C0 | proposed - known_time | -0.0037 | 0.0445 | 2/5 |
| C1 | proposed - periodic | +0.0132 | 0.0168 | 1/5 |
| C1 | proposed - drift_only | -0.0053 | 0.0050 | 4/5 |
| C1 | proposed - random | +0.0205 | 0.0104 | 0/5 |
| C1 | proposed - always_on | +0.0381 | 0.0142 | 0/5 |
| C1 | proposed - known_time | -0.0079 | 0.0193 | 3/5 |
| C2 | proposed - periodic | +0.0121 | 0.0248 | 1/5 |
| C2 | proposed - drift_only | +0.0003 | 0.0007 | 0/5 |
| C2 | proposed - random | +0.0177 | 0.0222 | 0/5 |
| C2 | proposed - always_on | +0.0366 | 0.0237 | 0/5 |
| C2 | proposed - known_time | -0.0368 | 0.0379 | 4/5 |

## Drift outcomes (means over seeds)

| dataset   | scenario   | method     |   detected |   n_events |   misses |   mean_detection_delay |   false_alarms_per_100_client_steps |   recovered |   censored |   mean_recovery_time |   expired_requests |   mean_alarm_to_service |   n_client_trainings |
|:----------|:-----------|:-----------|-----------:|-----------:|---------:|-----------------------:|------------------------------------:|------------:|-----------:|---------------------:|-------------------:|------------------------:|---------------------:|
| clinical  | C0         | drift_only |        0   |          0 |      0   |                  nan   |                               0.25  |           0 |          0 |              nan     |                  0 |                   0     |                 10   |
| clinical  | C0         | known_time |        0   |          0 |      0   |                  nan   |                               0.25  |           0 |          0 |              nan     |                  0 |                 nan     |                  0   |
| clinical  | C0         | proposed   |        0   |          0 |      0   |                  nan   |                               0.25  |           0 |          0 |              nan     |                  0 |                   0     |                 10   |
| clinical  | C1         | drift_only |        1.8 |          2 |      0.2 |                    7.9 |                               0.326 |           2 |          0 |               11.5   |                  0 |                   0.9   |                 26.2 |
| clinical  | C1         | known_time |        1.8 |          2 |      0.2 |                    7.9 |                               0.326 |           2 |          0 |                7.5   |                  0 |                 nan     |                 20   |
| clinical  | C1         | proposed   |        1.8 |          2 |      0.2 |                    7.9 |                               0.326 |           2 |          0 |               10.8   |                  0 |                   0.5   |                 26.2 |
| clinical  | C2         | drift_only |        2   |          3 |      1   |                    5.9 |                               0.511 |           3 |          0 |                9     |                  0 |                   0.167 |                 30.8 |
| clinical  | C2         | known_time |        2   |          3 |      1   |                    5.9 |                               0.511 |           3 |          0 |                8.067 |                  0 |                 nan     |                 30   |
| clinical  | C2         | proposed   |        2   |          3 |      1   |                    5.9 |                               0.511 |           3 |          0 |                9     |                  0 |                   0.167 |                 30.8 |

## Priority ablation

```
                                  post_logloss_pooled         post_logloss_worst_client         mean_alarm_to_service         expired_requests         bytes_adaptation              
                                                 mean     std                      mean     std                  mean     std             mean     std             mean           std
dataset  scenario method                                                                                                                                                             
clinical C0       proposed                     0.3381  0.0660                    0.6618  0.2610                0.0000  0.0000              0.0  0.0000        4122944.0  2.932912e+06
         C1       proposed                     0.3171  0.0228                    0.6277  0.1649                0.5000  0.7071              0.0  0.0000       10588166.4  3.951504e+06
                  proposed_nosev               0.3224  0.0234                    0.6375  0.1737                0.9000  0.9690              0.0  0.0000       10588166.4  3.951504e+06
                  proposed_nowait              0.3240  0.0417                    0.6292  0.1828                0.0400  0.0894              0.8  0.8367        9761120.0  3.765206e+06
         C2       proposed                     0.3167  0.0310                    0.6264  0.1889                0.1667  0.3727              0.0  0.0000       12647587.2  3.008850e+06
```

## Budget sweep

```
                                       post_logloss_pooled  bytes_adaptation  n_client_trainings
dataset  scenario method     budget_b                                                           
clinical C0       drift_only 0.25                   0.3381         4122944.0                10.0
                  periodic   0.25                   0.2927        20599360.0                50.0
                  proposed   0.25                   0.3381         4122944.0                10.0
         C1       drift_only 0.10                   0.3377         7058368.0                17.4
                             0.25                   0.3224        10588166.4                26.2
                             0.50                   0.3207        10426393.6                29.8
                  periodic   0.10                   0.3394         8239744.0                20.0
                             0.25                   0.3039        20599360.0                50.0
                             0.50                   0.2860        41199520.0               200.0
                  proposed   0.10                   0.3280         7058368.0                17.4
                             0.25                   0.3171        10588166.4                26.2
                             0.50                   0.3189        10426393.6                29.8
         C2       drift_only 0.25                   0.3164        12647587.2                30.8
                  periodic   0.25                   0.3046        20599360.0                50.0
                  proposed   0.25                   0.3167        12647587.2                30.8
```
