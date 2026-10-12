# AGENTS.md — Non-Negotiable Rules for All Agents and Contributors
#
# Every automated agent and every human contributor MUST follow these rules.
# If you want to deviate from any rule, STOP and ask the project owner first.
# These rules exist to protect the scientific validity of the results.

# Container Resource Forecasting — Agent Rules

## 1. Chronological Splits Only

- Splits are **chronological per series**: first 70 % → train, next 15 % → validation, last 15 % → test.
- **Never shuffle.** Never use `train_test_split` with `shuffle=True`.
- **Never random-split** a time series. The ordering must reflect real temporal order.

## 2. No Data Leakage

- **Scalers** (e.g. `StandardScaler`, `MinMaxScaler`) are **fit on train only**, then applied to val and test.
- **Thresholds, percentile limits** (static baseline) and **safety margins** are **fit/tuned on train/val only**.
- **Lag and rolling features** use only strictly past values: feature at time `t` uses only `t-1` and earlier. Never use `t` itself or any future value as a feature.
- **Features never cross container boundaries**: each container's features are computed from that container's own series only.
- **Targets are shifted correctly** for horizon `h`: the target at position `i` is the value at `i + h`. Verify this in tests.

## 3. Test Set is Sacred

- The **test set is touched once**, for final reporting only.
- **Hyperparameters** (e.g. number of lags, tree depth, number of estimators) are tuned on **validation only**.
- The **safety margin** for the proactive rule is tuned on **validation only**.
- After touching the test set, do not re-tune or re-train based on test results.

## 4. Reproducibility

- **Fix all random seeds** using `src/common.set_seed(seed)` at the top of every script and notebook.
- Every run **must save** config, seed, git commit hash and metrics via `src/common.save_run()`.
- Results without a saved run directory are not valid experimental results.

## 5. Honesty and Accuracy

- **Never invent, hardcode, or round numbers to look better.**
- **Never tune metrics to look better.** If something fails or performs poorly, report the real output.
- **Never claim code works without running it and showing the output.**

## 6. Testing

- Write `pytest` tests for **leakage** and **split correctness** for each data/feature module.
- Tests live in `tests/`. Run `pytest` before every commit.
- A passing test suite is required before merging any new module.

## 7. Data Interpretation

- Alibaba data contains **normalised utilisation** (values between 0 and 1). Results are in **relative units**.
- Replay limits are **hypothetical** — the Alibaba traces do not include real container limits.
- Do not compare normalised Alibaba results with raw Docker stats results without resampling to the same interval first.

## 8. Code Quality

- Keep modules **small and commented** so a student can explain each one.
- Each function must have a docstring describing its purpose, parameters and return value.
- No function longer than ~80 lines without a clear justification.

## 9. Version Control

- **Commit to git after each completed step** with a clear, descriptive message.
- The commit message must mention: what was done, which stage it belongs to, and whether tests pass.
- Example: `scaffold: add src/common.py and configs/default.yaml; tests pass`

## 10. Specification Adherence

- **Follow `docs/HANDOVER.md` at all times.** It is the primary specification.
- If you want to deviate from the spec, **stop and ask the project owner** before making any changes.
- If something in the spec is unclear or contradictory, **flag it explicitly** and wait for clarification.

## 11. Credit Economy

- Write each file once. Do not view files you just wrote. Do not re-read parts of AGENTS.md you do not need.
- No exploratory or scratch scripts and no testing of alternatives unless a test fails.
- Run pytest at most twice (once at the end, once more only after fixing failures). Run each validation run once.
- Never poll or wait on long jobs. Give the command and let the user run it.
- The final message contains only the items requested. No recap of what you did, no emojis, no tables of file links.

## 12. Results Schema

- Every model evaluation appends rows to `results/metrics_long.csv` with columns:
  `run_id, dataset, model, horizon_steps, target(cpu|mem), container_id, split, mae, rmse, smape, mase, train_time_s, pred_latency_ms, model_size_kb, n_models, n_jobs, fallback_rate, seed`.
- **Model Size (`model_size_kb`)**: Size of the pickled MINIMAL parameter payload needed to produce forecasts for ONE model instance (for per-container models, that container's own model; for global models, the single model), excluding training data and fit diagnostics:
  - `naive`: 0 KB.
  - `ma`: size of pickled integer `k`.
  - `ets`: size of pickled config tuple + fitted smoothing parameters and Box-Cox lambda.
  - `arima`: size of pickled order tuple `(p, d, q)`, trend flag string, and the estimated parameters array.
  - `rf`: the full fitted estimator pickled.
  - `xgb`: raw serialized bytes via `xgboost.Booster.save_raw`.
  - `lstm`: PyTorch `state_dict` serialized.
- `n_models`: Number of fitted model instances needed to cover all containers for that (model, target, horizon): 600 for per-container models, 1 for global models.
- `n_jobs`: Number of parallel workers used in fitting (e.g. 1 for sequential, or cpu count / specified n_jobs).
- Predictions for **both val and test** splits are saved to `results/predictions/<run_tag>/<dataset>_<model>_h<H>_<target>.parquet` with columns `container_id, origin_step, target_step, split, y_true, y_pred`. Val predictions are needed to tune the safety margin; test predictions are for final reporting.
- Never overwrite old runs; append. If a predictions file already exists, append with a new run_id so both sets are preserved.
- Tuning logs go to `results/tuning/`; best hyperparameters are saved to `results/tuning/best_params.json`; all tuning records are append-only with a run_id.
- Headline results are always on all 600 containers; tiers are for stratified reporting only.

## 13. Primary Dataset

- **Primary dataset:** `data/processed/alibaba_v2022_8h.parquet` (600 containers × 480 steps, dataset tag `alibaba8h`).
- The 4-hour file (`alibaba_v2022.parquet`) is used only for robustness / ablation checks, **not** for primary results.
- All metric rows and prediction files use dataset tag `alibaba8h` for the primary dataset.

## 14. Forecasting-Origin Rule

- A forecast made at **origin step t** may use only observations at steps **≤ t** (strictly no look-ahead).
- A supervised row belongs to the split of its **target step** (t + h), not its origin step t.
  - Train rows: target_step < train_end (= 336 for 480 steps).
  - Val rows: val_start ≤ target_step < val_end.
  - Test rows: test_start ≤ target_step < test_end.
- Val and test rows may legitimately use lag features drawn from earlier splits (the observation history before t is real past data).
- **Never** compute split membership from the origin step t; always use the target step t + h.
- Split bounds are always computed from `split_bounds(n_steps, fracs)` in `src/data/loader.py`. **Never hard-code** step numbers.

