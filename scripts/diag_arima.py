import pandas as pd, glob
for f in sorted(glob.glob("results/tuning/**/arima_*_choices.csv", recursive=True)):
    d = pd.read_csv(f); print(f, len(d), list(d.columns))
    c = [x for x in d.columns if "config" in x or "order" in x]
    if c: print(d[c[0]].value_counts().head(8).to_string())
def load(m, tgt):
    fs = [f for f in glob.glob(f"results/predictions/**/alibaba8h_{m}_h1_{tgt}.parquet", recursive=True) if "val20d" in f]
    return pd.read_parquet(fs[0]) if fs else None
for tgt in ["mem","cpu"]:
    a, n = load("arima", tgt), load("naive", tgt)
    if a is None or n is None: print("missing preds for", tgt); continue
    a, n = a[a.split=="test"], n[n.split=="test"]
    m = a.merge(n, on=["container_id","target_step"], suffixes=("_a","_n"))
    m["ea"] = (m.y_pred_a - m.y_true_a).abs(); m["en"] = (m.y_pred_n - m.y_true_a).abs()
    g = m.groupby("container_id")[["ea","en"]].mean(); g["ratio"] = g.ea/g.en
    print(tgt, "arima/naive MAE ratio per container:"); print(g.ratio.describe().round(3).to_string())
    print("share identical to naive (<1e-9):", float(((m.y_pred_a-m.y_pred_n).abs()<1e-9).mean()))
    print("worst 5:"); print(g.sort_values("ratio", ascending=False).head(5).round(5).to_string())
