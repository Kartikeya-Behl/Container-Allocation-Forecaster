import glob
from pathlib import Path
import pandas as pd
for f in sorted(glob.glob("results/tuning/trees/xgb_*_v2_trials.csv")):
    d = pd.read_csv(f); d = d[d["state"] == "COMPLETE"]
    b = d.loc[d["value"].idxmin()]
    print(Path(f).name, "| trials", len(d), "| best n_estimators", int(b["user_attrs_n_estimators"]),
          "| trials at cap:", int((d["user_attrs_n_estimators"] >= 2000).sum()))
