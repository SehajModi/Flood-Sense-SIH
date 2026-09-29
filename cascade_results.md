| Stage | Days flagged | Events caught | Recall | Precision | F1 | False alarms /1000 days |
|---|---|---|---|---|---|---|
| All days (no filter) | 37,701 (100.0%) | 359/359 | 100.0% | 1.0% | 0.02 | 990.5 |
| Tier 1: slave IF | 27,469 (72.9%) | 343/359 | 95.5% | 1.2% | 0.02 | 719.5 |
| + Tier 2: master CNN confirms | 11,451 (30.4%) | 324/359 | 90.3% | 2.8% | 0.05 | 295.1 |
| + Tier 3: cloud confirms | 6,061 (16.1%) | 304/359 | 84.7% | 5.0% | 0.09 | 152.7 |
| Ref: Tier 3 alone, same recall | 6,370 (16.9%) | 314/359 | 87.5% | 4.9% | 0.09 | 160.6 |

Test rows: 37,701; extreme = Rainfall >= 37.2 mm next day. Thresholds set on validation to keep 95%/90%/90% of events at tiers 1/2/3 (float CNN, not INT8).
