p = "scripts/trees.py"
s = open(p, encoding="utf-8").read()
old = "def latency_ms(m, kind, X, n=100):\n"
new = old + '    if kind == "xgb":\n        m.get_booster().set_param({"nthread": 1})\n    else:\n        m.set_params(n_jobs=1)\n'
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("patched")
