| Metric | Value | Source |
|---|---|---|
| **Tier 1: Isolation Forest (slave)** |  |  |
| Same-day extreme rain: ROC-AUC | 0.997 (AP 0.740 vs base 0.011) | measured |
| Next-day extreme rain: ROC-AUC | 0.899 (AP 0.180 vs base 0.011) | measured |
|   recall/precision/F1 @10% alert rate | 76.7% / 5.9% / 0.11 | measured |
| **Tier 2: 1D-CNN, float (master)** |  |  |
| Next-day extreme rain: ROC-AUC | 0.919 (AP 0.181 vs base 0.011) | measured |
|   recall/precision/F1 @5% alert rate | 59.4% / 13.0% / 0.21 | measured |
| **Tier 3: cloud, multi-city gradient boosting** |  |  |
| Cities in network | 10 | measured |
| Next-day extreme rain: ROC-AUC | 0.942 (AP 0.279 vs base 0.011) | measured |
|   recall/precision/F1 @5% alert rate | 70.1% / 15.4% / 0.25 | measured |
Dataset: 91,310 city-days, 2000-2024 daily weather, 10 Indian cities. NO pressure/humidity in this source -> features are rain_mm, rain_3d, temp_max, temp_min, gust_max. Extreme = rain >= 41.3 mm (99th pct).
