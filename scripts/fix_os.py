import ast
p = "src/run_forecast.py"
src = open(p, encoding="utf-8").read()
tree = ast.parse(src)
has_os = any(isinstance(n, ast.Import) and any(a.name == "os" for a in n.names) for n in ast.walk(tree))
if has_os:
    print("os already imported")
else:
    L = src.split("\n")
    idx = next(n.lineno - 1 for n in tree.body
               if isinstance(n, (ast.Import, ast.ImportFrom))
               and not (isinstance(n, ast.ImportFrom) and n.module == "__future__"))
    L.insert(idx, "import os")
    open(p, "w", encoding="utf-8").write("\n".join(L))
    print("inserted 'import os' at line", idx + 1)
ast.parse(open(p, encoding="utf-8").read()); print("syntax ok")
