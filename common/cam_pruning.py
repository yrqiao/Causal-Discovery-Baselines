"""CAM pruning (Bühlmann et al., 2014), shared by SCORE and DiffAN.

The R files in cam_pruning_R/ are the ones shipped with SCORE / DiffAN (taken from the CAM package).
Given a topological order encoded as a fully connected DAG, every edge whose GAM p-value is above
`cutoff` is removed.
"""
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from cdt.utils.R import launch_R_script

R_DIR = Path(__file__).resolve().parent / "cam_pruning_R"


def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d, d))
    for i, var in enumerate(top_order):
        A[var, top_order[i + 1:]] = 1
    return A


def np_to_csv(array, save_path):
    output = os.path.join(save_path, "tmp_" + str(uuid.uuid4()) + ".csv")
    pd.DataFrame(array).to_csv(output, header=False, index=False)
    return output


def cam_pruning(A, X, cutoff, verbose=False):
    """A: [d, d] candidate DAG, X: [n, d] numpy array. Returns the pruned [d, d] adjacency."""
    with tempfile.TemporaryDirectory() as save_path:
        arguments = {
            "{R_DIR}": str(R_DIR),
            "{PATH_DATA}": np_to_csv(np.asarray(X), save_path),
            "{PATH_DAG}": np_to_csv(A, save_path),
            "{PATH_RESULTS}": os.path.join(save_path, "results.csv"),
            "{CUTOFF}": str(cutoff),
            "{VERBOSE}": "TRUE" if verbose else "FALSE",
        }

        def retrieve_result():
            return pd.read_csv(arguments["{PATH_RESULTS}"]).values

        return launch_R_script(str(R_DIR / "cam_pruning.R"), arguments, output_function=retrieve_result)
