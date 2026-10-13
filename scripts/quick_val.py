import pandas as pd
df = pd.read_csv("results/metrics_long.csv")
print("rows:", len(df), "| cols:", len(df.columns))
d = df[(df["split"]=="val") & (df["horizon_steps"].isin([1,5]))]
print(d.groupby(["target","horizon_steps","model"])[["mae","mase","fallback_rate"]].mean().round(5).to_string())
print(d.groupby(["model","target"])[["train_time_s","pred_latency_ms","model_size_kb","n_models"]].mean().round(3).to_string())
print(d.groupby(["model","target"]).size().to_string())
