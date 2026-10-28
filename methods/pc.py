"""PC algorithm (Spirtes et al., 2000) from pgmpy."""
import numpy as np
from pgmpy.estimators import PC


def run(dataset, cfg, workdir):
    pc = PC(dataset.df)
    model = pc.estimate(return_type=cfg["return_type"], significance_level=cfg["significance_level"],
                        ci_test=cfg["ci_test"], variant=cfg["variant"],
                        max_cond_vars=cfg["max_cond_vars"], show_progress=False)
    print(model.edges())

    index = {name: i for i, name in enumerate(dataset.columns)}
    adj = np.zeros((len(index), len(index)), dtype=int)
    for u, v in model.edges():
        adj[index[u], index[v]] = 1
    return adj
