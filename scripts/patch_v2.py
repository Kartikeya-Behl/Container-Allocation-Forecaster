p = "scripts/trees.py"
s = open(p, encoding="utf-8").read()
def rep(old, new, n):
    global s
    assert s.count(old) == n, (old, s.count(old))
    s = s.replace(old, new)
rep('[3, 5, 10, 20]', '[3, 5, 10, 20, 30, 45, 60]', 2)
rep('tr.suggest_int("max_depth", 2, 8)', 'tr.suggest_int("max_depth", 2, 12)', 1)
rep('n_estimators=1000, early_stopping_rounds=50', 'n_estimators=2000, early_stopping_rounds=50', 1)
rep('name = f"{model}_{tgt}_h{h}"', 'name = f"{model}_{tgt}_h{h}_v2"', 1)
rep('    done = len([t for t in study.trials if t.state.name == "COMPLETE"])\n',
    '    if not study.trials and f"{model}/{tgt}/h{h}" in best:\n        study.enqueue_trial(best[f"{model}/{tgt}/h{h}"]["params"])\n    done = len([t for t in study.trials if t.state.name == "COMPLETE"])\n', 1)
open(p, "w", encoding="utf-8").write(s)
print("patched")
