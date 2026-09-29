| Metric | Value | Source |
|---|---|---|
| **Tier 1: Isolation Forest (slave node)** |  |  |
| Config | 50 trees, max depth 8, 5 features | measured |
| Total split/leaf nodes | 5,496 | measured |
| Flash (model) | ~38 KB | estimate: 7 B/node |
| RAM at inference | 20 B features + <100 B locals, no heap | by design |
| Compares per inference (worst case) | 400 | measured |
| Latency, laptop CPU | 2 us/sample (batched) | measured, NOT MCU |
| Latency, ESP32-S3 @240MHz | ~83 us | ESTIMATE, verify on HW |
| C export matches sklearn | max abs score error 1.5e-07 | measured |
| Same-day extreme rain: ROC-AUC | 0.998  (AP 0.811 vs base 0.010) | measured |
|   recall/precision/F1 @5% alert rate | 100.0% / 15.1% / 0.26 | measured |
|   recall/precision/F1 @10% alert rate | 100.0% / 8.5% / 0.16 | measured |
| Next-day extreme rain: ROC-AUC | 0.833  (AP 0.106 vs base 0.010) | measured |
|   recall/precision/F1 @10% alert rate | 58.5% / 4.9% / 0.09 | measured |
| **Tier 2: tiny 1D-CNN, INT8 (master node)** |  |  |
| Parameters | 209 | measured |
| TFLite INT8 size | 4.5 KB | measured |
| Tensor RAM (upper bound) | 0.4 KB | measured, no reuse |
| Latency, laptop CPU | 10 us/sample | measured, NOT MCU |
| Next-day extreme rain: ROC-AUC (INT8) | 0.885  (AP 0.096 vs base 0.009) | measured |
|   recall/precision/F1 @5% alert rate | 52.2% / 9.1% / 0.15 | measured |
| **Tier 3: cloud, multi-station fusion (AWS tier)** |  |  |
| Stations in network | 45 | measured |
| Next-day, single-station GBM: ROC-AUC | 0.922  (AP 0.195 vs base 0.010) | measured |
|   recall/precision/F1 @5% alert rate | 63.5% / 12.1% / 0.20 | measured |
| Next-day, multi-station cloud GBM: ROC-AUC | 0.920  (AP 0.172 vs base 0.010) | measured |
|   recall/precision/F1 @5% alert rate | 58.5% / 11.1% / 0.19 | measured |
| Gain from network fusion (ROC-AUC) | -0.002 | measured |
Dataset rows used: 125,732. 'Extreme rain' = Rainfall >= 37.2 mm (99th pct). Train on the first 70% of dates (normal days only for IF), test on the last 30%. Daily data: 9am->3pm pressure change is the pressure-gradient proxy.
