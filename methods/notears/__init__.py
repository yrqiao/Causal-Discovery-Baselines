"""NOTEARS (Zheng et al., 2018) and its nonlinear MLP version (Zheng et al., 2020).
Source: https://github.com/xunzheng/notears"""
import numpy as np

from common.utils import set_deterministic
from .linear import notears_linear
from .nonlinear import NotearsMLP, notears_nonlinear


def run(dataset, cfg, workdir):
    set_deterministic(cfg["seed"])
    X = dataset.X.astype(np.float32)

    if cfg["model"] == "linear":
        W_est = notears_linear(X, lambda1=cfg["lambda1"], loss_type=cfg["loss_type"], max_iter=cfg["max_iter"],
                               h_tol=cfg["h_tol"], rho_max=cfg["rho_max"], w_threshold=cfg["w_threshold"])
    elif cfg["model"] == "nonlinear":
        model = NotearsMLP(dims=[X.shape[1], cfg["hidden"], 1], bias=True)
        W_est = notears_nonlinear(model, X, lambda1=cfg["lambda1"], lambda2=cfg["lambda2"], max_iter=cfg["max_iter"],
                                  h_tol=cfg["h_tol"], rho_max=cfg["rho_max"], w_threshold=cfg["w_threshold"])
    else:
        raise ValueError(f"unknown NOTEARS model: {cfg['model']}")

    print(W_est)
    return (W_est != 0).astype(int)
