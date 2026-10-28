import torch

from .stein import SCORE


def run(dataset, cfg, workdir):
    X = torch.tensor(dataset.X)
    A_SCORE, top_order_SCORE = SCORE(X, cfg["eta_G"], cfg["eta_H"], cfg["cam_cutoff"],
                                     normalize_var=cfg["normalize_var"], dispersion=cfg["dispersion"],
                                     pruning=cfg["pruning"], threshold=cfg["threshold"])
    print(f"topological order: {[dataset.columns[i] for i in top_order_SCORE]}")
    print(A_SCORE)
    return A_SCORE.astype(int)
