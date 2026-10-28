"""Dataset registry: reads configs/datasets.yaml and loads the selected columns of each CSV."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASETS_YAML = ROOT / "configs" / "datasets.yaml"


@dataclass
class Dataset:
    name: str
    df: pd.DataFrame        # only the selected columns, in config order
    graph: np.ndarray       # ground truth, graph[i, j] = 1 means columns[i] -> columns[j]

    @property
    def columns(self):
        return list(self.df.columns)

    @property
    def X(self):
        return self.df.to_numpy(dtype=np.float64)


def dataset_names():
    with open(DATASETS_YAML) as f:
        return list(yaml.safe_load(f))


def load_dataset(name):
    with open(DATASETS_YAML) as f:
        spec = yaml.safe_load(f)[name]
    raw = pd.read_csv(DATA_DIR / spec["file"])
    if raw.columns.duplicated().any():
        raise ValueError(f"{spec['file']} has duplicated column names")
    df = raw[spec["columns"]]
    graph = np.asarray(spec["graph"], dtype=int)
    assert graph.shape == (df.shape[1], df.shape[1]), f"{name}: graph shape does not match columns"
    return Dataset(name=name, df=df, graph=graph)
