"""scripts/trees.py - tune + fit + predict global RF / XGBoost. Run from the project root."""
import argparse, json, os, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from data.loader import split_bounds            # noqa: E402
from features.windows import make_supervised    # noqa: E402

warnings.filterwarnings("ignore")
DATA = "data/processed/alibaba_v2022_8h.parquet"
DATASET, SEED, HORIZONS = "alibaba8h", 42, [1, 2, 3, 4, 5]
N_TRIALS = {"xgb": 40, "rf": 20}


def load(n_containers=None):
    df = pd.read_parquet(DATA).sort_values(["container_id", "step"])
    ids = sorted(df["container_id"].unique())
    if n_containers:
        ids = sorted(np.random.RandomState(SEED).choice(ids, n_containers, replace=False).tolist())
        df = df[df["container_id"].isin(ids)]
    n_steps = int(df["step"].nunique())
    Ys = {t: df.pivot(index="container_id", columns="step", values=t).loc[ids].to_numpy(dtype=float)
          for t in ("cpu", "mem")}
    return ids, Ys, n_steps


def leakage_checks(Y, ids, bounds):
    (_, tr_e), (va_s, va_e), (te_s, te_e) = bounds
    m = make_supervised(Y, 1, 5, "train", bounds, container_ids=ids)[4]
    assert m.target_step.max() < tr_e, "train targets leak past train_end"
    for sp, (a, b) in (("val", (va_s, va_e)), ("test", (te_s, te_e))):
        m = make_supervised(Y, 5, 10, sp, bounds, container_ids=ids)[4]
        assert m.target_step.min() >= a and m.target_step.max() < b, f"{sp} targets out of range"
    cut = te_s + 30
    Yc = Y.copy()
    Yc[:, cut:] = np.random.RandomState(0).rand(Y.shape[0], Y.shape[1] - cut)
    X1, _, _, _, m1 = make_supervised(Y, 5, 10, "test", bounds, container_ids=ids)
    X2, _, _, _, m2 = make_supervised(Yc, 5, 10, "test", bounds, container_ids=ids)
    keep = (m1.origin_step < cut).to_numpy()
    assert keep.sum() > 0 and np.array_equal(X1[keep], X2[keep]), "features use future values"
    print("leakage checks passed", flush=True)


class Cache:
    """Supervised train/val arrays per n_lags for one (target, horizon)."""
    def __init__(self, Y, ids, bounds, h):
        self.Y, self.ids, self.bounds, self.h, self.c = Y, ids, bounds, h, {}

    def __call__(self, nl):
        if nl not in self.c:
            self.c[nl] = {sp: make_supervised(self.Y, self.h, nl, sp, self.bounds,
                                              container_ids=self.ids)[:4] for sp in ("train", "val")}
        return self.c[nl]


def fit_xgb(p, Xtr, ytr, Xva, yva):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, eval_metric="mae",
                         tree_method="hist", n_jobs=-1, random_state=SEED, **p)
    t0 = time.perf_counter()
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m, time.perf_counter() - t0


def fit_rf(p, Xtr, ytr):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_jobs=-1, random_state=SEED, **p)
    t0 = time.perf_counter()
    m.fit(Xtr, ytr)
    return m, time.perf_counter() - t0


def predict(m, kind, X):
    if kind == "xgb":
        return m.predict(X, iteration_range=(0, m.best_iteration + 1))
    return m.predict(X)


def size_kb(m, kind):
    raw = m.get_booster()[: m.best_iteration + 1].save_raw() if kind == "xgb" else pickle.dumps(m)
    return len(raw) / 1024


def latency_ms(m, kind, X, n=100):
    idx = np.random.RandomState(SEED).randint(0, len(X), n)
    ts = []
    for i in idx:
        t0 = time.perf_counter()
        predict(m, kind, X[i:i + 1])
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def trial_xgb(tr, D):
    nl = tr.suggest_categorical("n_lags", [3, 5, 10, 20])
    p = dict(
        learning_rate=tr.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=tr.suggest_int("max_depth", 2, 8),
        min_child_weight=tr.suggest_int("min_child_weight", 1, 20),
        subsample=tr.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=tr.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=tr.suggest_float("reg_lambda", 0.1, 10, log=True),
        reg_alpha=tr.suggest_float("reg_alpha", 0.0, 1.0),
        objective=tr.suggest_categorical("objective", ["reg:squarederror", "reg:absoluteerror"]))
    d = D(nl)
    m, ft = fit_xgb(p, d["train"][0], d["train"][1], d["val"][0], d["val"][1])
    tr.set_user_attr("fit_time_s", ft)
    tr.set_user_attr("n_estimators", int(m.best_iteration) + 1)
    return float(np.mean(np.abs(predict(m, "xgb", d["val"][0]) - d["val"][1])))


def trial_rf(tr, D):
    nl = tr.suggest_categorical("n_lags", [3, 5, 10, 20])
    p = dict(
        n_estimators=tr.suggest_categorical("n_estimators", [100, 200, 300]),
        max_depth=tr.suggest_categorical("max_depth", [6, 10, 14, 20]),
        min_samples_leaf=tr.suggest_categorical("min_samples_leaf", [10, 20, 50, 100]),
        max_features=tr.suggest_categorical("max_features", [0.33, 0.6, 1.0]))
    d = D(nl)
    m, ft = fit_rf(p, d["train"][0], d["train"][1])
    tr.set_user_attr("fit_time_s", ft)
    return float(np.mean(np.abs(m.predict(d["val"][0]) - d["val"][1])))


def tune_one(model, tgt, h, Y, ids, bounds, n_trials, out, best, best_path):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    name = f"{model}_{tgt}_h{h}"
    study = optuna.create_study(
        study_name=name, direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
        storage="sqlite:///" + (out / "optuna.db").as_posix(), load_if_exists=True)
    done = len([t for t in study.trials if t.state.name == "COMPLETE"])
    D = Cache(Y, ids, bounds, h)
    obj = (lambda tr: trial_xgb(tr, D)) if model == "xgb" else (lambda tr: trial_rf(tr, D))

    def cb(st, tr):
        print(f"[{name}] trial {tr.number:3d}  val_mae={tr.value:.6f}  best={st.best_value:.6f}  "
              f"({tr.duration.total_seconds():.0f}s)", flush=True)

    if n_trials > done:
        study.optimize(obj, n_trials=n_trials - done, callbacks=[cb])
    comp = [t for t in study.trials if t.state.name == "COMPLETE"]
    bt = study.best_trial
    best[f"{model}/{tgt}/h{h}"] = {
        "params": bt.params, "n_estimators": bt.user_attrs.get("n_estimators"),
        "val_mae": bt.value, "n_trials": len(comp),
        "tuning_time_s": float(sum(t.duration.total_seconds() for t in comp))}
    best_path.write_text(json.dumps(best, indent=2))
    study.trials_dataframe().to_csv(out / f"{name}_trials.csv", index=False)
    print(f"[{name}] DONE best val_mae={bt.value:.6f} params={bt.params}", flush=True)


def final_one(model, tgt, Y, ids, bounds, best, tune_hs, tag):
    out = Path(f"results/predictions/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    fit_total, lat_total, sizes, used = 0.0, 0.0, {}, {}
    for h in HORIZONS:
        src = min(tune_hs, key=lambda x: (abs(x - h), x))
        p = dict(best[f"{model}/{tgt}/h{src}"]["params"])
        nl = p.pop("n_lags")
        S = {sp: make_supervised(Y, h, nl, sp, bounds, container_ids=ids) for sp in ("train", "val", "test")}
        if model == "xgb":
            m, ft = fit_xgb(p, S["train"][0], S["train"][1], S["val"][0], S["val"][1])
        else:
            m, ft = fit_rf(p, S["train"][0], S["train"][1])
        frames = []
        for sp in ("val", "test"):
            X, yd, yl, ylast, meta = S[sp]
            pred = ylast + predict(m, model, X)
            df = meta[["container_id", "origin_step", "target_step", "split"]].copy()
            df["y_true"], df["y_pred"] = yl, pred
            frames.append(df)
            if sp == "val":
                val_mae = float(np.mean(np.abs(pred - yl)))
        pd.concat(frames, ignore_index=True).to_parquet(
            out / f"{DATASET}_{model}_h{h}_{tgt}.parquet", index=False)
        fit_total += ft
        lat_total += latency_ms(m, model, S["test"][0])
        sizes[h] = size_kb(m, model)
        used[h] = {"params_from_h": src, "n_lags": nl, **p}
        print(f"[final {model}/{tgt}/h{h}] fit={ft:.1f}s  val_mae={val_mae:.6f}  size={sizes[h]:.0f}KB", flush=True)
        del m, S
    tj = Path(f"results/trees_timing_{tag}.json")
    T = json.loads(tj.read_text()) if tj.exists() else {}
    recs = [best[f"{model}/{tgt}/h{h}"] for h in tune_hs]
    T[f"{model}/{tgt}"] = {
        "train_time_s": fit_total, "latency_ms_5step": lat_total, "model_size_kb_by_h": sizes,
        "n_trials": int(sum(r["n_trials"] for r in recs)),
        "tuning_time_s": float(sum(r["tuning_time_s"] for r in recs)),
        "n_jobs": os.cpu_count(), "params_by_h": used}
    tj.write_text(json.dumps(T, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "final", "all"], default="all")
    ap.add_argument("--models", nargs="+", default=["xgb", "rf"])
    ap.add_argument("--targets", nargs="+", default=["cpu", "mem"])
    ap.add_argument("--tune-horizons", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="full600")
    a = ap.parse_args()

    ids, Ys, n_steps = load(20 if a.quick else None)
    bounds = split_bounds(n_steps)
    print("containers:", len(ids), "| steps:", n_steps, "| bounds:", bounds, flush=True)
    leakage_checks(Ys["cpu"], ids, bounds)
    tag = "quick" if a.quick else a.tag
    out = Path("results/tuning/trees" + ("_quick" if a.quick else ""))
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "best_params.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else {}

    if a.stage in ("tune", "all"):
        for model in a.models:
            for tgt in a.targets:
                for h in a.tune_hs if hasattr(a, "tune_hs") else a.tune_horizons:
                    n = a.trials or (5 if a.quick else N_TRIALS[model])
                    tune_one(model, tgt, h, Ys[tgt], ids, bounds, n, out, best, best_path)
    if a.stage in ("final", "all"):
        best = json.loads(best_path.read_text())
        for model in a.models:
            for tgt in a.targets:
                final_one(model, tgt, Ys[tgt], ids, bounds, best, a.tune_horizons, tag)


if __name__ == "__main__":
    main()


