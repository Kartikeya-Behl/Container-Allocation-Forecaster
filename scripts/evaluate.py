"""scripts/evaluate.py - score every model from saved predictions with ONE metric implementation."""
import glob, sys
from pathlib import Path
import numpy as np
import pandas as pd

TAG = sys.argv[1] if len(sys.argv) > 1 else "full600"
DATASET = "alibaba8h"
meta = pd.read_parquet("data/processed/container_meta_8h.parquet").set_index("container_id")


def per_container(df, tgt):
    e = (df["y_pred"] - df["y_true"]).abs()
    t = pd.DataFrame({"cid": df["container_id"].to_numpy(), "e": e.to_numpy(), "e2": (e ** 2).to_numpy(),
                      "s": (2 * e / np.maximum(df["y_true"].abs() + df["y_pred"].abs(), 1e-3)).to_numpy()})
    g = t.groupby("cid")
    r = pd.DataFrame({"mae": g["e"].mean(), "rmse": np.sqrt(g["e2"].mean()), "smape": g["s"].mean()})
    r["mase"] = r["mae"] / np.maximum(meta.loc[r.index, f"naive_mae1_{tgt}"].to_numpy(), 1e-3)
    r.index.name = "container_id"
    return r.reset_index()


rows, seen = [], set()
for f in sorted(glob.glob(f"results/predictions/{TAG}/{DATASET}_*_h*_*.parquet")):
    p = Path(f).stem.split("_")
    model, h, tgt = p[1], int(p[2][1:]), p[3]
    if (model, h, tgt) in seen:
        print("DUPLICATE prediction file, skipped:", f)
        continue
    seen.add((model, h, tgt))
    d = pd.read_parquet(f)
    for sp in ("val", "test"):
        r = per_container(d[d["split"] == sp], tgt)
        r.insert(0, "split", sp); r.insert(0, "horizon_steps", h)
        r.insert(0, "target", tgt); r.insert(0, "model", model)
        rows.append(r)

M = pd.concat(rows, ignore_index=True)
M.to_csv(f"results/metrics_eval_{TAG}.csv", index=False)
key = ["model", "target", "horizon_steps", "split"]
print("wrote", len(M), "rows | models:", sorted(M["model"].unique()))
print("containers per group (min,max):", M.groupby(key).size().agg(["min", "max"]).tolist())

S = pd.read_csv("results/metrics_long.csv")
S = S[S["run_id"].astype(str).str.startswith(TAG + "_")]
cols = ["mae", "rmse", "smape", "mase"]
j = M.groupby(key)[cols].mean().join(S.groupby(key)[cols].mean(), how="inner", lsuffix="_mine", rsuffix="_stored")
print("groups compared with stored metrics:", len(j))
for c in cols:
    d = (j[c + "_mine"] - j[c + "_stored"]).abs() / j[c + "_stored"].abs().clip(lower=1e-12)
    print(f"{c:6s} max relative diff = {d.max():.2e}  worst group: {d.idxmax()}")
