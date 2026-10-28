#!/usr/bin/env bash
# Run every method on all datasets (inside the activated env):  bash scripts/run_all.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results/logs
for method in cam pc dag_gnn gran_dag notears diffan score; do
    echo "=== ${method} (log: results/logs/${method}.log)"
    python run.py --method "${method}" 2>&1 | tee "results/logs/${method}.log" | tail -n 25
done
