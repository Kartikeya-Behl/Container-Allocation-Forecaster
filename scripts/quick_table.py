import pandas as pd
df = pd.read_csv("results/metrics_long.csv")
print("columns:", list(df.columns)); print("rows:", len(df))
if "run_tag" in df.columns: df = df[df["run_tag"]=="val20d"]
k = [c for c in ["model","target","split","horizon_steps"] if c in df.columns]
print("rows per group (expect 20 containers):"); print(df.groupby(k).size().groupby(["model","target"]).agg(["min","max"]).to_string())
t = df[(df["split"]=="test") & (df["horizon_steps"].isin([1,5]))]
cols = [c for c in ["mae","rmse","smape","mase"] if c in df.columns]
print(t.groupby(["target","horizon_steps","model"])[cols].mean().round(5).to_string())
for c in ["fallback_rate","train_time_s","pred_latency_ms","model_size_kb"]:
    if c in df.columns: print(c); print(df.groupby(["model","target"])[c].mean().round(4).to_string())
