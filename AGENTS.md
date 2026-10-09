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
