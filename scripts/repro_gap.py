import pandas as pd
a = pd.read_csv("results/metrics_eval_full600_run1.csv"); b = pd.read_csv("results/metrics_eval_full600.csv")
j = a.merge(b, on=["model","target","horizon_steps","split","container_id"], suffixes=("_1","_2"))
print("rows", len(a), len(b), "merged", len(j))
for c in ["mae","rmse","smape","mase"]:
    print(c, "max abs diff between runs:", (j[c+"_1"]-j[c+"_2"]).abs().max())
g = b[b.horizon_steps.isin([1,5])].groupby(["target","horizon_steps","model","split"])["mase"].mean().unstack("split")
g["test_minus_val"] = g["test"] - g["val"]
print(g.round(4).to_string())
