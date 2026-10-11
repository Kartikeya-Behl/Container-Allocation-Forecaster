# Audit Report: Alibaba microservices-v2022 Processed Dataset

**Dataset Path:** `data\processed\alibaba_v2022.parquet`  
**Dimensions:** 600 containers x 240 steps = 144,000 observations (60s sampling interval)  
**Split Definitions:** Train (steps 0–167, 70%), Val (steps 168–203, 15%), Test (steps 204–239, 15%)

## 1. Distribution by Split

| Metric | Split | Mean | Std | Median | P5 | P95 |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | Train (0-167) | 0.172666 | 0.140279 | 0.123834 | 0.034810 | 0.464401 |
| CPU | Val (168-203) | 0.179836 | 0.144885 | 0.129460 | 0.037562 | 0.468123 |
| CPU | Test (204-239) | 0.178394 | 0.144723 | 0.126995 | 0.036774 | 0.469246 |
| MEM | Train (0-167) | 0.547680 | 0.186462 | 0.573305 | 0.151558 | 0.811536 |
| MEM | Val (168-203) | 0.546880 | 0.186230 | 0.574287 | 0.152530 | 0.814211 |
| MEM | Test (204-239) | 0.547400 | 0.186012 | 0.574819 | 0.153077 | 0.813613 |


## 2. Train-to-Test Distribution Shift per Container

| Metric | Median Shift (std) | P90 Shift (std) | P99 Shift (std) | Test Max > Train Max | Test Min < Train Min |
| --- | --- | --- | --- | --- | --- |
| CPU | 0.5038 | 1.4413 | 3.4763 | 19.00% (114/600) | 14.17% (85/600) |
| MEM | 0.8529 | 2.9382 | 22.5453 | 41.83% (251/600) | 9.50% (57/600) |


## 3. Cross-Sectional Mean Series & Synchronized Regime Events

- **Typical Step-to-Step Std of Average CPU:** `0.002676`
- **Typical Step-to-Step Std of Average Mem:** `0.000555`
- **Synchronized Regime Events (> 3*std step-change in global mean):**
  - CPU jump steps: [29, 119, 179]
  - Memory jump steps: [60, 81]


## 4. Saturation and Zero Utilisation

| Metric | Values >= 0.99 (%) | Values == 0.0 (%) | Containers with >10% at cap |
| --- | --- | --- | --- |
| CPU | 0.002% (3) | 0.000% (0) | 0 / 600 (0.00%) |
| MEM | 0.153% (221) | 0.000% (0) | 1 / 600 (0.17%) |


## 5. Flat Runs and Zero-Change Steps

| Metric | Max Flat Run (across all) | Median Longest Run | P90 Longest Run | Containers with Run > 20 steps | Zero-Change Steps Overall |
| --- | --- | --- | --- | --- | --- |
| CPU | 3 | 1 | 2 | 0 (0.00%) | 0.19% (277) |
| MEM | 2 | 1 | 1 | 0 (0.00%) | 0.04% (63) |


## 6. Quantization and Smallest Step Sizes

| Metric | Unique Values Min | Unique Values P25 | Unique Values Median | Unique Values P75 | Unique Values Max | Smallest Non-Zero Step |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | 104 | 201 | 219 | 231 | 240 | 1.05e-12 |
| MEM | 206 | 238 | 239 | 240 | 240 | 6.18e-08 |


## 7. Memory Restarts / Large Relative Jumps (> 20%)

- **Containers with >= 1 jump > 20%:** `69 / 600` (11.50%)
- **Total Jump Events Across Dataset:** `835`

### First 10 Jump Events:

| container_id | step | val_before | val_after | pct_change |
| --- | --- | --- | --- | --- |
| MS_10984_POD_0 | 58 | 0.429261 | 0.166123 | -61.30% |
| MS_10984_POD_0 | 59 | 0.166123 | 0.285654 | +71.95% |
| MS_10984_POD_0 | 60 | 0.285654 | 0.351824 | +23.16% |
| MS_11440_POD_987 | 229 | 0.618423 | 0.044991 | -92.72% |
| MS_11440_POD_987 | 230 | 0.044991 | 0.359087 | +698.14% |
| MS_11440_POD_987 | 231 | 0.359087 | 0.457089 | +27.29% |
| MS_1162_POD_9 | 97 | 0.752578 | 0.453166 | -39.78% |
| MS_1162_POD_9 | 98 | 0.453166 | 0.263988 | -41.75% |
| MS_1162_POD_9 | 99 | 0.263988 | 0.423953 | +60.60% |
| MS_1162_POD_9 | 106 | 0.569708 | 0.180393 | -68.34% |


## 8. Spikes (Max |First Difference| per Container)

| Metric | Min | P25 | Median | P75 | P90 | P99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU | 0.007563 | 0.039614 | 0.074552 | 0.187055 | 0.369952 | 0.752623 | 0.867988 |
| MEM | 0.000724 | 0.002021 | 0.003017 | 0.006314 | 0.129151 | 0.587623 | 0.785366 |


## 9. Forecastability (Train Steps 0–167 Only)

| Metric | AC(1) Median | AC(1) P10 | AC(1) P90 | AC(5) Median | AC(5) P10 | AC(5) P90 | Share AC(1) < 0.3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU | 0.6808 | 0.1620 | 0.9383 | 0.3646 | -0.0039 | 0.8553 | 17.33% (104/600) |
| MEM | 0.9045 | 0.4581 | 0.9752 | 0.6768 | 0.1361 | 0.8862 | 4.67% (28/600) |


## 10. Within-Container CPU–Memory Correlation (Train Steps)

| Min | P10 | P25 | Median | P75 | P90 | Max | Share Corr < 0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -0.9604 | -0.4063 | -0.0720 | 0.1539 | 0.3840 | 0.6403 | 0.9694 | 30.83% |


## 11. Cross-Container Near-Duplicates (Train CPU Series)

| Total Container Pairs | Pairs with Pearson r > 0.95 | Pairs Sharing Same msname | Pairs Across Different Services |
| --- | --- | --- | --- |
| 179,700 | 141 (0.078%) | 0 (0.00%) | 141 |


## Flags

- **Test-set extrapolation risk**: In 19.0% of containers for CPU and 41.8% for memory, the test maximum strictly exceeds the maximum seen during training. Static percentile baselines fit on train alone will systematically under-provision on these series.
- **White-noise / low-predictability series**: 17.33% of containers exhibit train lag-1 autocorrelation < 0.3 for CPU utilisation, indicating near-uncorrelated noise where autoregressive point forecasts will struggle against moving averages.
- **Cross-container collinearity**: 141 container pairs have Pearson correlation r > 0.95 on train CPU. 0 of those pairs belong to the exact same microservice, while 141 occur across different microservices, indicating cluster-wide scheduled cron tasks or shared daemon rhythms.

