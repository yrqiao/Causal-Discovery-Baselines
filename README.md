# Causal Discovery Baselines

Seven causal discovery methods evaluated on 19 datasets (7 physical scenes, 11 hypothetical
structures, waterflow), with one shared environment and one command per method.

| Method   | `--method` | Implementation                                                          | Code                |
|----------|------------|-------------------------------------------------------------------------|---------------------|
| CAM      | `cam`      | [cdt](https://github.com/FenTechSolutions/CausalDiscoveryToolbox) 0.6.0 wrapper of the R package CAM 1.0 | `methods/cam.py`    |
| PC       | `pc`       | [pgmpy](https://github.com/pgmpy/pgmpy) 0.1.26, Pearson-correlation CI test | `methods/pc.py`     |
| DAG-GNN  | `dag_gnn`  | [ronikobrosly/DAG_from_GNN](https://github.com/ronikobrosly/DAG_from_GNN) | `methods/dag_gnn/`  |
| GraN-DAG | `gran_dag` | [kurowasan/GraN-DAG](https://github.com/kurowasan/GraN-DAG)              | `methods/gran_dag/` |
| NOTEARS  | `notears`  | [xunzheng/notears](https://github.com/xunzheng/notears) (linear and MLP) | `methods/notears/`  |
| DiffAN   | `diffan`   | [vios-s/DiffAN](https://github.com/vios-s/DiffAN)                        | `methods/diffan/`   |
| SCORE    | `score`    | [paulrolland1307/SCORE](https://github.com/paulrolland1307/SCORE)        | `methods/score/`    |

## Setup

Requires conda and an NVIDIA driver supporting CUDA 12.1 (only DiffAN uses the GPU; everything else runs on CPU).

```bash
bash scripts/setup_env.sh          # creates the conda env `causal_baselines` (Python 3.11 + R 4.4)
conda activate causal_baselines
```

`setup_env.sh` does three things:
1. `conda env create -f environment.yml` (Python packages, R, and the R packages available on conda),
2. `Rscript scripts/install_r_packages.R` (mboost, pcalg, SID from CRAN; CAM 1.0 from the CRAN archive),
3. `python scripts/check_env.py` (imports everything and runs CAM and SID once end to end).

R is used by CAM, by the CAM pruning step of SCORE and DiffAN, and by the SID metric.

## Run

One command per method; each runs all 19 datasets and evaluates them:

```bash
python run.py --method cam
python run.py --method pc
python run.py --method dag_gnn
python run.py --method gran_dag
python run.py --method notears
python run.py --method diffan
python run.py --method score
```

Other options:

```bash
python run.py --method notears --datasets pendulum magnet   # a subset of datasets
python run.py --list                                        # list methods and datasets
bash scripts/run_all.sh                                     # every method, logs in results/logs/
```

### Outputs

```
results/<method>/metrics.csv                         one row per dataset (and variant)
results/<method>/adjacency/<dataset>[__<variant>].csv  predicted adjacency, A[i, j] = 1 means i -> j
results/<method>/work/<dataset>/                     method-specific artifacts (GraN-DAG models and plots)
```

Running a subset of datasets only replaces those rows in `metrics.csv`. DiffAN reports two rows per dataset
(`variant` = `residue` / `masking`); its `time_sec` is the total for both.

### Metrics

Computed by `common/metrics.py` on the d x d adjacency matrices (diagonal included), as in the original notebooks:
`accuracy`, `precision`, `tpr`, `fpr`, `f1`, `shd` (number of differing entries, so a reversed edge counts twice)
and `sid` (Structural Intervention Distance, `cdt.metrics.SID(target=truth, pred=prediction)`).

## Configuration

- `configs/datasets.yaml`: for each dataset, the CSV file in `data/`, the columns used (in order) and the
  ground-truth graph over those columns. To add a dataset, drop the CSV into `data/` and add an entry.
- `configs/<method>.yaml`: hyper-parameters. `default` applies to every dataset, `datasets.<name>` overrides
  it for one dataset (only NOTEARS uses per-dataset settings).

## Using your own data

1. Put a CSV file in `data/`, one row per sample and one column per variable, with a header row of unique
   column names. The columns used for causal discovery must be numeric with no missing values; any other
   columns (ids, image paths, ...) can stay in the file and are ignored.

   ```
   id,x1,x2,y,img_path
   0,0.53,1.20,2.71,0.png
   1,-0.12,0.88,1.05,1.png
   ...
   ```

2. Add an entry to `configs/datasets.yaml`:

   ```yaml
   my_dataset:              # the name passed to --datasets
     file: my_data.csv      # file in data/
     columns: [x1, x2, y]   # variables to use, in this order
     graph:                 # ground truth over `columns`: graph[i][j] = 1 means columns[i] -> columns[j]
       - [0, 0, 1]          # x1 -> y
       - [0, 0, 1]          # x2 -> y
       - [0, 0, 0]
   ```

   `graph` must be a square 0/1 matrix of size `len(columns)`, and acyclic (SID is only defined for DAGs).
   At least 2 variables are needed.

3. Run any method on it:

   ```bash
   python run.py --method notears --datasets my_dataset
   ```

New datasets use the `default` block of `configs/<method>.yaml`. To tune a method for one dataset, add an
override under `datasets:` in that file, e.g. in `configs/notears.yaml`:

```yaml
datasets:
  my_dataset: {model: linear, lambda1: 0.1}
```

SCORE's memory grows with n² (about 20 GB for n = 15000, d = 3), so subsample very large datasets for it.

## Runtime

All 19 datasets, measured with the 7 methods running at the same time on 20 CPU cores + one RTX 4090:

| CAM   | PC     | DAG-GNN | GraN-DAG | NOTEARS | DiffAN (GPU) | SCORE  |
|-------|--------|---------|----------|---------|--------------|--------|
| 3 min | < 1 min| 50 min  | 40 min   | 13 min  | 40 min       | 40 min |

SCORE builds n x n kernel matrices on the CPU and needs ~20 GB of RAM on the 15000-sample datasets.

## Layout

```
run.py                 entry point
configs/               datasets.yaml + one yaml per method
data/                  the 19 CSV files
common/                data loading, metrics, seeding, CAM pruning (R) shared by SCORE and DiffAN
methods/<method>/      method code; each exposes run(dataset, cfg, workdir) -> adjacency
scripts/               environment setup / check, run_all.sh
```
