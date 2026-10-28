"""GraN-DAG (Lachapelle et al., ICLR 2020). Source: https://github.com/kurowasan/GraN-DAG

The original code reads data{i}.npy / DAG{i}.npy from a folder and writes everything under an experiment
folder; this wrapper prepares that folder for one dataset and returns <exp>/to-dag/DAG.npy.
"""
import argparse
import shutil

import numpy as np
import torch

from .main import main


def run(dataset, cfg, workdir):
    data_dir, exp_dir = workdir / "data", workdir / "exp"
    for d in (data_dir, exp_dir):
        shutil.rmtree(d, ignore_errors=True)  # GraN-DAG skips stages whose outputs already exist
    data_dir.mkdir(parents=True)

    np.save(data_dir / "data1.npy", dataset.X)
    # The ground truth is only used by GraN-DAG for logging during training.
    np.save(data_dir / "DAG1.npy", dataset.graph)
    np.save(data_dir / "CPDAG1.npy", dataset.graph)

    opt = argparse.Namespace(**cfg, exp_path=str(exp_dir), data_path=str(data_dir), i_dataset="1",
                             num_vars=dataset.X.shape[1])
    try:
        main(opt)
    finally:
        torch.set_default_tensor_type(torch.FloatTensor)  # main() switches the global default to double

    return np.load(exp_dir / "to-dag" / "DAG.npy").astype(int)
