"""DiffAN (Sanchez et al., ICLR 2023): topological ordering with diffusion models, followed by CAM pruning.
Source: https://github.com/vios-s/DiffAN"""
from common.utils import set_deterministic
from .diffan import DiffAN
from .utils import num_errors

VARIANTS = {
    "residue": dict(residue=True),    # "with Deciduous Residue"
    "masking": dict(residue=False),   # "without Deciduous Residue / Masking only"
}


def run(dataset, cfg, workdir):
    # As in the original notebook: seed once, then run the variants one after another.
    set_deterministic(cfg["seed"])
    X = dataset.X
    results = {}
    for name in cfg["variants"]:
        print(f"Run DiffAN ({name})")
        diffan = DiffAN(X.shape[1], epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                        learning_rate=cfg["learning_rate"], **VARIANTS[name])
        adj_matrix, order = diffan.fit(X)
        print(f"topological order: {[dataset.columns[i] for i in order]}, "
              f"order errors: {num_errors(order, dataset.graph)}")
        print(adj_matrix)
        results[name] = adj_matrix.astype(int)
    return results
