import importlib.resources
import json
from typing import Callable, Dict

import numpy as np
import pyvista as pv

import cnn_tpms


def compute_density_thickness_models(degree: int = 9) -> Dict[str, Callable]:
    json_filename = (
        importlib.resources.files(cnn_tpms) / "data" / "thickness_densities.json"
    ).as_posix()
    with open(json_filename) as json_file:
        dataset = json.load(json_file)
    models: Dict[str, Callable] = {}
    for key, value in dataset.items():
        points = np.asarray(value)
        thickness, densities = points[:, 0], points[:, 1]
        p = np.polyfit(densities, thickness, degree)
        model = np.poly1d(p)
        models[key] = model
    return models


def convert_binary_voxel_grid_to_pyvista(voxel_grid: np.ndarray) -> pv.PolyData:
    wrapped_grid = pv.wrap(voxel_grid.astype(float))
    return wrapped_grid.threshold(1)
