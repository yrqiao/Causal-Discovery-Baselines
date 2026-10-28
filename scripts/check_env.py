"""Sanity checks for the shared environment: Python packages, CUDA, R packages, and a
tiny end-to-end call of every R-backed component (CAM, SID)."""
import importlib
import shutil
import subprocess
import sys

import numpy as np

failed = []


def check(name, fn):
    try:
        info = fn()
        print(f"  [ok]   {name}" + (f"  ({info})" if info else ""))
    except Exception as e:  # noqa: BLE001
        failed.append(name)
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")


print("Python packages")
for mod in ["numpy", "scipy", "pandas", "sklearn", "networkx", "torch", "igraph",
            "statsmodels", "matplotlib", "seaborn", "yaml", "tqdm", "pgmpy", "cdt"]:
    check(mod, lambda m=mod: getattr(importlib.import_module(m), "__version__", ""))

print("CUDA")
import torch  # noqa: E402

check("torch.cuda", lambda: f"available={torch.cuda.is_available()}, "
                           f"devices={torch.cuda.device_count()}, cuda={torch.version.cuda}")
check("torch.func", lambda: importlib.import_module("torch.func").jacrev and "")

print("R")
rscript = shutil.which("Rscript")
check("Rscript on PATH", lambda: rscript if rscript and sys.prefix in rscript
      else (_ for _ in ()).throw(RuntimeError(f"{rscript} is not inside {sys.prefix}")))
for pkg in ["CAM", "SID", "mgcv", "mboost", "glmnet", "pcalg"]:
    check(f"R package {pkg}", lambda p=pkg: subprocess.run(
        ["Rscript", "-e", f"cat(as.character(packageVersion('{p}')))"],
        check=True, capture_output=True, text=True).stdout)

print("End-to-end")


def tiny_sid():
    from cdt.metrics import SID
    truth = np.array([[0, 1], [0, 0]])
    return f"SID(truth, truth)={int(SID(truth, truth))}, SID(truth, reversed)={int(SID(truth, truth.T))}"


def tiny_cam():
    import pandas as pd
    from cdt.causality.graph import CAM
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 300)
    y = np.sin(x) * 2 + 0.2 * rng.normal(size=300)
    g = CAM().predict(pd.DataFrame({"x": x, "y": y}))
    return f"edges={list(g.edges())}"


check("cdt SID", tiny_sid)
check("cdt CAM", tiny_cam)

if failed:
    print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
    sys.exit(1)
print("\nEnvironment OK.")
