import json
import multiprocessing
import time
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyvista as pv
from fire import Fire
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from cnn_tpms.data_generation import surfaces
from cnn_tpms.data_generation.lattice_generator import _compute_mesh_grid


def compute_density_thickness_polycurves_from_dataset_file(
    json_filename: str,
) -> Dict[str, Callable]:
    with open(json_filename) as json_file:
        dataset = json.load(json_file)
    models: Dict[str, Callable] = {}
    for key, value in dataset.items():
        points = np.asarray(value)
        thickness, densities = points[:, 0], points[:, 1]
        degree = 9
        p = np.polyfit(densities, thickness, degree)
        model = np.poly1d(p)
        models[key] = model
    return models


def generate_tpms(
    density: float,
    tpms_array: npt.ArrayLike,
    mesh_grid_xyz: Tuple[np.ndarray, np.ndarray, np.ndarray],
    structure_name: str,
    folder_name: Path,
    density_thickness_models: Dict[str, Callable],
) -> Tuple[str, float]:
    surface_grid = pv.StructuredGrid(*mesh_grid_xyz)
    surface_offset: float = density_thickness_models[structure_name](density)
    surface_grid["lower_surface"] = (tpms_array - surface_offset / 2.0).ravel(
        order="F"
    )  # type: ignore
    surface_grid["upper_surface"] = (tpms_array + surface_offset / 2.0).ravel(
        order="F"
    )  # type: ignore

    sheet = surface_grid.clip_scalar(
        scalars="upper_surface", invert=False
    ).clip_scalar(scalars="lower_surface")
    density = abs(sheet.volume)
    filename = structure_name + f"_{str(surface_offset).replace('.', '_')}.npz"
    shape_filename = folder_name.joinpath(Path(filename)).as_posix()

    np.savez(shape_filename, points=sheet.points)
    return shape_filename, density


def main(
    n_samples_per_class: int = 1000,
    mesh_resolution: int = 100,
    dataset_folder: str = "dataset_mesh/",
) -> None:
    start_time = time.perf_counter()
    surface_functions = [
        surfaces.gyroid,
        surfaces.schwarz_p,
        surfaces.schwarz_d,
        surfaces.neovius,
        surfaces.schoen_iwp,
        surfaces.schoen_frd,
        surfaces.fischer_koch_s,
        surfaces.pmy,
        surfaces.honeycomb,
    ]
    dataset_folder = Path(dataset_folder)

    # Global structure parameters
    repeat_cell = (1, 1, 1)
    cell_size = (1, 1, 1)
    # Dataset generation parameters
    seed = 69
    print(f"Dataset folder {dataset_folder}")
    print(f"Mesh resolution {mesh_resolution}")
    print(f"Cell size {cell_size}")
    print(f"Repeat cell {repeat_cell}")
    print(f"Samples per class {n_samples_per_class}")
    print(f"Gen seed {seed}")

    random_generator = np.random.default_rng(seed=seed)
    density_thickness_models: Dict[str, Callable] = (
        compute_density_thickness_polycurves_from_dataset_file(
            "thickness_densities.json"
        )
    )
    # Precomputing random densities per class
    densities_per_class: List[npt.ArrayLike] = [
        random_generator.uniform(low=0.01, size=n_samples_per_class)
        for _ in surface_functions
    ]

    # Compute reusable mesh grid data
    mesh_grid_xyz: Tuple[np.ndarray, np.ndarray, np.ndarray] = (
        _compute_mesh_grid(
            repeat_cell=repeat_cell,
            unit_cell_size=cell_size,
            mesh_resolution=mesh_resolution,
        )
    )
    k_x, k_y, k_z = np.divide(2 * np.pi, cell_size)
    n_cores = multiprocessing.cpu_count()
    total_filenames: List[str] = []
    total_densities: List[float] = []
    loop = tqdm(zip(surface_functions, densities_per_class), leave=True)
    for surface_function, densities in loop:
        filenames_and_densities: List[Tuple[List[str], List[float]]] = []
        # Generate base surface
        print(surface_function)
        structure_name = str(surface_function).split(" ")[1]
        folder_name = dataset_folder.joinpath(structure_name)
        folder_name.mkdir(parents=True, exist_ok=True)
        tpms_grid = surface_function(
            k_x * mesh_grid_xyz[0],
            k_y * mesh_grid_xyz[1],
            k_z * mesh_grid_xyz[2],
        )
        # Compute offsets parallel
        function = partial(
            generate_tpms,
            tpms_array=tpms_grid,
            mesh_grid_xyz=mesh_grid_xyz,
            structure_name=structure_name,
            folder_name=folder_name,
            density_thickness_models=density_thickness_models,
        )

        with multiprocessing.Pool(processes=n_cores) as pool:
            filenames_and_densities = pool.map(function, densities)
        print("Generation done")
        for filenames, densities in filenames_and_densities:
            total_filenames.append(filenames)
            total_densities.append(densities)

        # Save Data

    dataset_mapping = {
        "filename": total_filenames,
        "density": total_densities,
    }
    dataset_dataframe = pd.DataFrame(dataset_mapping)
    label_mapping_filename = str(dataset_folder.joinpath(Path("mapping.json")))
    # Add class column to dataframe
    dataset_dataframe["class"] = list(
        map(
            lambda filename: filename.split("/")[1],
            dataset_dataframe["filename"],
        )
    )
    classes_unique = sorted(dataset_dataframe["class"].unique())
    name_label_mapping = {
        name_class: i for i, name_class in enumerate(classes_unique)
    }
    dataset_dataframe["label"] = [
        name_label_mapping[class_name]
        for class_name in dataset_dataframe["class"]
    ]
    # Save dataframe and json mapping
    # Split data
    train_dataframe, test_dataframe = train_test_split(
        dataset_dataframe,
        test_size=0.3,
        stratify=dataset_dataframe["label"],
        random_state=seed,
    )
    # Save split for reproductibility
    csv_train_dataset_filename = dataset_folder.joinpath(
        Path("train_dataset.csv")
    )
    csv_test_dataset_filename = dataset_folder.joinpath(
        Path("test_dataset.csv")
    )
    train_dataframe.to_csv(csv_train_dataset_filename, index=False)
    test_dataframe.to_csv(csv_test_dataset_filename, index=False)
    csv_dataset_filename = dataset_folder.joinpath(Path("dataset.csv"))
    with open(label_mapping_filename, "w") as fp:
        json.dump(name_label_mapping, fp)
    end_time = time.perf_counter()
    print(
        f"Generated {len(total_filenames)} samples in {end_time - start_time} seconds"
    )


if __name__ == "__main__":
    Fire(main)
