"""scripts/compare.py - final comparison tables, tests and figures. Reads saved results only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = "full600"
OUT, FIG = Path("results/comparison"), Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
M = pd.read_csv(f"results/metrics_eval_{TAG}.csv")
meta = pd.read_parquet("data/processed/container_meta_8h.parquet")
ORDER = ["naive", "ma", "ets", "arima", "rf", "xgb"]
H = sorted(M["horizon_steps"].unique())
T, V = M[M.split == "test"], M[M.split == "val"]
pd.set_option("display.width", 200)

print("== sanity: predictions must differ across adjacent horizons ==")
for model in ORDER:
    for tgt in ("cpu", "mem"):
        prev = None
        for h in H:
            d = pd.read_parquet(f"results/predictions/{TAG}/alibaba8h_{model}_h{h}_{tgt}.parquet")
            d = d[d.split == "test"].sort_values(["container_id", "target_step"])
            if prev is not None and len(prev) == len(d):
                if np.abs(prev.y_pred.to_numpy() - d.y_pred.to_numpy()).max() < 1e-12:
                    print("  IDENTICAL predictions:", model, tgt, "h", h - 1, "vs", h)
            prev = d
print("  sanity pass complete")

agg = T.groupby(["target", "horizon_steps", "model"])[["mae", "rmse", "smape", "mase"]].mean().reset_index()
agg.to_csv(OUT / "test_metrics_mean.csv", index=False)
for tgt in ("cpu", "mem"):
    for col in ("mase", "mae"):
        print(f"\n=== TEST {tgt}: mean {col.upper()} by horizon (lower is better) ===")
        print(agg[agg.target == tgt].pivot(index="model", columns="horizon_steps", values=col)
              .reindex(ORDER).round(5).to_string())

vm = V.groupby(["target", "horizon_steps", "model"])["mase"].mean().reset_index()
champ = vm.loc[vm.groupby(["target", "horizon_steps"])["mase"].idxmin()].set_index(["target", "horizon_steps"])["model"]
print("\n== champion per (target, horizon), chosen on VALIDATION MASE ==")
print(champ.unstack().to_string())


def piv(tgt, h):
    d = T[(T.target == tgt) & (T.horizon_steps == h)]
    return d.pivot(index="container_id", columns="model", values="mase")


def wp(d):
    if np.allclose(d, 0):
        return np.nan
    try:
        return wilcoxon(d).pvalue
    except ValueError:
        return np.nan


def family(rows):
    df = pd.DataFrame(rows)
    p = df["p_raw"].fillna(1.0).to_numpy()
    df["p_holm"] = multipletests(p, method="holm")[1]
    return df


vs_naive, vs_champ = [], []
for tgt in ("cpu", "mem"):
    for h in H:
        P = piv(tgt, h)
        rows = []
        for m in ORDER[1:]:
            d = (P[m] - P["naive"]).to_numpy()
            rows.append(dict(target=tgt, h=h, model=m, rel_improve_pct=100 * (1 - P[m].mean() / P["naive"].mean()),
                             win_rate=float((d < 0).mean()), p_raw=wp(d)))
        vs_naive.append(family(rows))
        c = champ[(tgt, h)]
        rows = []
        for m in ORDER:
            if m == c:
                continue
            d = (P[c] - P[m]).to_numpy()
            rows.append(dict(target=tgt, h=h, champion=c, other=m,
                             rel_improve_pct=100 * (1 - P[c].mean() / P[m].mean()),
                             win_rate=float((d < 0).mean()), p_raw=wp(d)))
        vs_champ.append(family(rows))
vs_naive, vs_champ = pd.concat(vs_naive), pd.concat(vs_champ)
vs_naive.to_csv(OUT / "wilcoxon_vs_naive.csv", index=False)
vs_champ.to_csv(OUT / "wilcoxon_champion_vs_others.csv", index=False)
print("\n== each model vs naive (per-container MASE; rel_improve>0 means better than naive; Holm over 5 models) ==")
print(vs_naive.round(4).to_string(index=False))
print("\n== champion vs each other model ==")
print(vs_champ.round(4).to_string(index=False))

tiers = T.merge(meta[["container_id", "mem_tier", "cpu_tier"]], on="container_id")
tiers["tier"] = np.where(tiers.target == "mem", tiers.mem_tier, tiers.cpu_tier.astype(str))
tt = tiers[tiers.horizon_steps.isin([1, 5])].groupby(["target", "tier", "horizon_steps", "model"])["mase"].mean().reset_index()
tt.to_csv(OUT / "test_mase_by_tier.csv", index=False)
for tgt in ("cpu", "mem"):
    print(f"\n=== TEST {tgt}: mean MASE by tier (rows) x model, h=1 and h=5 ===")
    print(tt[tt.target == tgt].pivot_table(index=["tier", "horizon_steps"], columns="model", values="mase")[ORDER].round(4).to_string())
print("\ncontainers per tier:", meta.groupby("mem_tier").size().to_dict(), meta.groupby("cpu_tier").size().to_dict())

S = pd.read_csv("results/metrics_long.csv")
S = S[S.run_id.astype(str).str.startswith(TAG + "_")]
cost = []
for (model, tgt), g in S.groupby(["model", "target"]):
    cost.append(dict(model=model, target=tgt, train_time_s=g.train_time_s.mean(), tuning_time_s=0.0,
                     latency_ms_5step=g.pred_latency_ms.mean(), size_kb_per_model=g.model_size_kb.mean(),
                     n_models=g.n_models.mean(), footprint_kb=g.model_size_kb.mean() * g.n_models.mean()))
TJ = json.load(open(f"results/trees_timing_{TAG}.json"))
for k, v in TJ.items():
    model, tgt = k.split("/")
    sz = list(v["model_size_kb_by_h"].values())
    cost.append(dict(model=model, target=tgt, train_time_s=v["train_time_s"], tuning_time_s=v["tuning_time_s"],
                     latency_ms_5step=v["latency_ms_5step"], size_kb_per_model=float(np.mean(sz)),
                     n_models=len(sz), footprint_kb=float(np.sum(sz))))
cost = pd.DataFrame(cost)
cost["model"] = pd.Categorical(cost["model"], ORDER)
cost = cost.sort_values(["target", "model"])
cost.to_csv(OUT / "cost_table.csv", index=False)
print("\n=== cost table (ETS/ARIMA/MA train_time includes their validation-based selection; trees: tuning_time separate) ===")
print(cost.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, tgt in zip(ax, ("cpu", "mem")):
    for m in ORDER:
        d = agg[(agg.target == tgt) & (agg.model == m)]
        a.plot(d.horizon_steps, d.mae, marker="o", label=m)
    a.set_title(f"{tgt}: test MAE vs horizon"); a.set_xlabel("horizon (min)"); a.set_ylabel("MAE"); a.grid(alpha=.3)
ax[0].legend(); plt.tight_layout(); plt.savefig(FIG / "mae_vs_horizon.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, tgt in zip(ax, ("cpu", "mem")):
    for i, h in enumerate((1, 5)):
        d = agg[(agg.target == tgt) & (agg.horizon_steps == h)].set_index("model").reindex(ORDER)
        a.bar(np.arange(6) + 0.38 * i, d.mase, 0.38, label=f"h={h}")
    a.set_xticks(np.arange(6) + 0.19); a.set_xticklabels(ORDER); a.set_title(f"{tgt}: test MASE"); a.grid(axis="y", alpha=.3)
ax[0].legend(); plt.tight_layout(); plt.savefig(FIG / "mase_bars.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, tgt in zip(ax, ("cpu", "mem")):
    for m in ORDER:
        c = cost[(cost.target == tgt) & (cost.model == m)].iloc[0]
        y = agg[(agg.target == tgt) & (agg.horizon_steps == 5) & (agg.model == m)].mase.iloc[0]
        a.scatter(max(c.train_time_s, 1e-3), y); a.annotate(m, (max(c.train_time_s, 1e-3), y))
    a.set_xscale("log"); a.set_xlabel("train time (s, log)"); a.set_ylabel("test MASE, h=5"); a.set_title(tgt); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "accuracy_vs_cost.png", dpi=150); plt.close()
print("\nwrote tables to", OUT, "and figures to", FIG)
