"""Run one causal discovery method on the benchmark datasets and evaluate it.

    python run.py --method notears                          # all 19 datasets
    python run.py --method pc --datasets pendulum magnet    # a subset
    python run.py --list                                    # available methods / datasets

Outputs (under --output, default results/):
    <method>/adjacency/<dataset>[__<variant>].csv   predicted adjacency (row -> column), with variable names
    <method>/metrics.csv                            one row per dataset (and variant)
"""
import argparse
import copy
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    # pgmpy's PC orients the undirected edges of the CPDAG while iterating over sets of variable names,
    # so with Python's per-process string hashing the orientation changes from run to run.
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.data import dataset_names, load_dataset  # noqa: E402
from common.metrics import evaluate  # noqa: E402

METHODS = ["cam", "pc", "dag_gnn", "gran_dag", "notears", "diffan", "score"]
METRIC_COLUMNS = ["accuracy", "precision", "tpr", "fpr", "f1", "shd", "sid"]


def method_config(method, dataset):
    with open(ROOT / "configs" / f"{method}.yaml") as f:
        spec = yaml.safe_load(f)
    cfg = copy.deepcopy(spec.get("default", {}))
    cfg.update((spec.get("datasets") or {}).get(dataset, {}))
    return cfg


def run_one(method, module, name, out_dir):
    dataset = load_dataset(name)
    cfg = method_config(method, name)
    workdir = out_dir / "work" / name
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"\n===== {method} | {name} | n={len(dataset.df)}, d={len(dataset.columns)} =====")
    print(f"config: {cfg}")

    start = time.time()
    result = module.run(dataset, cfg, workdir)
    elapsed = time.time() - start

    variants = result if isinstance(result, dict) else {"": result}
    rows = []
    for variant, adj in variants.items():
        stem = f"{name}__{variant}" if variant else name
        pd.DataFrame(adj, index=dataset.columns, columns=dataset.columns).to_csv(out_dir / "adjacency" / f"{stem}.csv")
        row = {"dataset": name, "variant": variant, "n_vars": len(dataset.columns),
               **evaluate(adj, dataset.graph), "time_sec": round(elapsed, 1)}
        print("  " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--datasets", nargs="+", default=["all"], help="dataset names, or 'all' (default)")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--list", action="store_true", help="list methods and datasets, then exit")
    args = parser.parse_args()

    if args.list:
        print("methods :", " ".join(METHODS))
        print("datasets:", " ".join(dataset_names()))
        return
    if args.method is None:
        parser.error("--method is required")

    names = dataset_names() if args.datasets == ["all"] else args.datasets
    unknown = sorted(set(names) - set(dataset_names()))
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")

    module = importlib.import_module(f"methods.{args.method}")
    out_dir = args.output / args.method
    (out_dir / "adjacency").mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"

    failed = []
    for name in names:
        try:
            rows = run_one(args.method, module, name, out_dir)
        except Exception:  # keep going with the other datasets
            traceback.print_exc()
            failed.append(name)
            continue
        # merge into metrics.csv so that partial runs (--datasets ...) update only their rows
        old = pd.read_csv(metrics_path, keep_default_na=False) if metrics_path.exists() else pd.DataFrame()
        if not old.empty:
            old = old[old["dataset"] != name]
        new = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
        order = {n: i for i, n in enumerate(dataset_names())}
        new = new.sort_values(by=["dataset", "variant"], key=lambda s: s.map(order) if s.name == "dataset" else s)
        new.to_csv(metrics_path, index=False)

    if metrics_path.exists():
        summary = pd.read_csv(metrics_path, keep_default_na=False)
        summary = summary[summary["dataset"].isin(names)]
        print(f"\n===== {args.method}: results saved to {metrics_path} =====")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if failed:
        print(f"\nFAILED on {len(failed)} dataset(s): {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
