"""CAM (Bühlmann, Peters & Ernest, 2014) through cdt, which calls the R package CAM."""
import networkx as nx
import numpy as np
from cdt.causality.graph import CAM


def run(dataset, cfg, workdir):
    cam = CAM(score=cfg["score"], cutoff=cfg["cutoff"], variablesel=cfg["variablesel"],
              selmethod=cfg["selmethod"], pruning=cfg["pruning"], prunmethod=cfg["prunmethod"],
              njobs=cfg["njobs"], verbose=cfg["verbose"])
    output = cam.predict(dataset.df)
    return (nx.to_numpy_array(output, nodelist=dataset.columns) != 0).astype(int)
