# Audit Report: Alibaba microservices-v2022 Processed Dataset

**Dataset Path:** `data\processed\alibaba_v2022_8h.parquet`  
**Dimensions:** 600 containers x 480 steps = 288,000 observations (60s sampling interval)  
**Split Definitions:** Train (steps 0–335, 70%), Val (steps 336–407, 15%), Test (steps 408–479, 15%)

## 1. Distribution by Split

| Metric | Split | Mean | Std | Median | P5 | P95 |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | Train (0-335) | 0.162592 | 0.127903 | 0.124380 | 0.033125 | 0.434633 |
| CPU | Val (336-407) | 0.165138 | 0.130894 | 0.126000 | 0.032873 | 0.457645 |
| CPU | Test (408-479) | 0.156049 | 0.124346 | 0.118440 | 0.030917 | 0.414377 |
| MEM | Train (0-335) | 0.544543 | 0.177698 | 0.564600 | 0.153905 | 0.806174 |
| MEM | Val (336-407) | 0.544200 | 0.177881 | 0.565413 | 0.151100 | 0.806531 |
| MEM | Test (408-479) | 0.545024 | 0.178000 | 0.564007 | 0.150193 | 0.807647 |


## 2. Train-to-Test Distribution Shift per Container

| Metric | Median Shift (std) | P90 Shift (std) | P99 Shift (std) | Test Max > Train Max | Test Min < Train Min |
| --- | --- | --- | --- | --- | --- |
| CPU | 0.5303 | 2.1560 | 4.9750 | 15.00% (90/600) | 36.33% (218/600) |
| MEM | 1.1172 | 2.9615 | 5.8785 | 49.00% (294/600) | 6.83% (41/600) |


## 3. Cross-Sectional Mean Series & Synchronized Regime Events

- **Typical Step-to-Step Std of Average CPU:** `0.002412`
- **Typical Step-to-Step Std of Average Mem:** `0.000456`
- **Synchronized Regime Events (> 3*std step-change in global mean):**
  - CPU jump steps: [59, 119, 179, 239, 269, 299, 359, 419]
  - Memory jump steps: [81, 148, 164, 247, 249, 272, 291, 321, 324, 477]


## 4. Saturation and Zero Utilisation

| Metric | Values >= 0.99 (%) | Values == 0.0 (%) | Containers with >10% at cap |
| --- | --- | --- | --- |
| CPU | 0.002% (5) | 0.000% (0) | 0 / 600 (0.00%) |
| MEM | 0.000% (0) | 0.000% (0) | 0 / 600 (0.00%) |


## 5. Flat Runs and Zero-Change Steps

| Metric | Max Flat Run (across all) | Median Longest Run | P90 Longest Run | Containers with Run > 20 steps | Zero-Change Steps Overall |
| --- | --- | --- | --- | --- | --- |
| CPU | 3 | 2 | 2 | 0 (0.00%) | 0.21% (604) |
| MEM | 2 | 1 | 2 | 0 (0.00%) | 0.05% (144) |


## 6. Quantization and Smallest Step Sizes

| Metric | Unique Values Min | Unique Values P25 | Unique Values Median | Unique Values P75 | Unique Values Max | Smallest Non-Zero Step |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | 147 | 355 | 407 | 444 | 480 | 1.05e-12 |
| MEM | 415 | 473 | 477 | 478 | 480 | 8.83e-09 |


## 7. Memory Restarts / Large Relative Jumps (> 20%)

- **Containers with >= 1 jump > 20%:** `92 / 600` (15.33%)
- **Total Jump Events Across Dataset:** `1,577`

### First 10 Jump Events:

| container_id | step | val_before | val_after | pct_change |
| --- | --- | --- | --- | --- |
| MS_10984_POD_0 | 58 | 0.429261 | 0.166123 | -61.30% |
| MS_10984_POD_0 | 59 | 0.166123 | 0.285654 | +71.95% |
| MS_10984_POD_0 | 60 | 0.285654 | 0.351824 | +23.16% |
| MS_12498_POD_2 | 14 | 0.473783 | 0.282366 | -40.40% |
| MS_12498_POD_2 | 15 | 0.282366 | 0.010119 | -96.42% |
| MS_12498_POD_2 | 16 | 0.010119 | 0.202054 | +1896.83% |
| MS_12498_POD_2 | 378 | 0.341802 | 0.001364 | -99.60% |
| MS_12498_POD_2 | 379 | 0.001364 | 0.096694 | +6987.53% |
| MS_12498_POD_2 | 380 | 0.096694 | 0.241718 | +149.98% |
| MS_12498_POD_2 | 477 | 0.312350 | 0.196330 | -37.14% |


## 8. Spikes (Max |First Difference| per Container)

| Metric | Min | P25 | Median | P75 | P90 | P99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU | 0.006210 | 0.045772 | 0.088028 | 0.208045 | 0.379963 | 0.756974 | 0.867988 |
| MEM | 0.000629 | 0.002067 | 0.003223 | 0.008698 | 0.265972 | 0.613164 | 0.767549 |


## 9. Forecastability (Train Steps 0–335 Only)

| Metric | AC(1) Median | AC(1) P10 | AC(1) P90 | AC(5) Median | AC(5) P10 | AC(5) P90 | Share AC(1) < 0.3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU | 0.7234 | 0.2157 | 0.9402 | 0.4252 | 0.0411 | 0.8721 | 13.67% (82/600) |
| MEM | 0.9421 | 0.6639 | 0.9907 | 0.7705 | 0.2892 | 0.9544 | 2.00% (12/600) |


## 10. Within-Container CPU–Memory Correlation (Train Steps)

| Min | P10 | P25 | Median | P75 | P90 | Max | Share Corr < 0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.8793 | -0.3505 | -0.0504 | 0.1697 | 0.3959 | 0.6108 | 0.9055 | 30.67% |


## 11. Cross-Container Near-Duplicates (Train CPU Series)

| Total Container Pairs | Pairs with Pearson r > 0.95 | Pairs Sharing Same msname | Pairs Across Different Services |
| --- | --- | --- | --- |
| 179,700 | 26 (0.014%) | 1 (3.85%) | 25 |


## Flags

- **Test-set extrapolation risk**: In 15.0% of containers for CPU and 49.0% for memory, the test maximum strictly exceeds the maximum seen during training. Static percentile baselines fit on train alone will systematically under-provision on these series.
- **White-noise / low-predictability series**: 13.67% of containers exhibit train lag-1 autocorrelation < 0.3 for CPU utilisation, indicating near-uncorrelated noise where autoregressive point forecasts will struggle against moving averages.

