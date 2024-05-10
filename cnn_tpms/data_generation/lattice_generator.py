from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import numpy as np
import numpy.typing as npt
import pyvista as pv

SurfaceFunction = Callable[[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike], npt.ArrayLike]


@dataclass
class TriplyPeriodicStructure:
    sheet: pv.PolyData
    upper_skeletal: pv.PolyData
    lower_skeletal: pv.PolyData


def _compute_mesh_grid(
    mesh_resolution: int,
    unit_cell_size: Tuple[float, float, float] = (1, 1, 1),
    repeat_cell: Tuple[int, int, int] = (1, 1, 1),
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    center_offset = 0.5  # Used to center structure on (0,0,0)
    linspaces: List[np.ndarray] = []
    for repeat_cell_axis, unit_cell_size_axis in zip(repeat_cell, unit_cell_size):
        start = -center_offset * unit_cell_size_axis * repeat_cell_axis
        end = center_offset * unit_cell_size_axis * repeat_cell_axis
        resolution = mesh_resolution * repeat_cell_axis
        linspaces.append(np.linspace(start, end, resolution))
    grid_x, grid_y, grid_z = np.meshgrid(*linspaces)
    return grid_x, grid_y, grid_z


def _repeat_cell(mesh: pv.PolyData, repeat_cell: Tuple[int, int, int]) -> pv.PolyData:
    if repeat_cell != (1, 1, 1):
        meshes_to_merge: list[pv.PolyData] = []
        x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
        dim_x = abs(x_min) + abs(x_max)
        dim_y = abs(y_min) + abs(y_max)
        dim_z = abs(z_min) + abs(z_max)
        for i_x in range(repeat_cell[0]):
            for i_y in range(repeat_cell[1]):
                for i_z in range(repeat_cell[2]):
                    new_mesh = mesh.copy()
                    new_mesh.translate(
                        (
                            -dim_x * (0.5 * repeat_cell[0] - 0.5 - i_x),
                            -dim_y * (0.5 * repeat_cell[1] - 0.5 - i_y),
                            -dim_z * (0.5 * repeat_cell[2] - 0.5 - i_z),
                        ),
                        inplace=True,
                    )
                    meshes_to_merge.append(new_mesh)
        mesh.merge(meshes_to_merge, inplace=True)
    return mesh


def _generate_tpms_parts(
    grid: pv.StructuredGrid,
    tpms_scalar_field: npt.ArrayLike,
    surface_offset: float,
    repeat_cell: Tuple[int, int, int] = (1, 1, 1),
) -> TriplyPeriodicStructure:
    grid["lower_surface"] = (tpms_scalar_field - surface_offset / 2.0).ravel(order="F")  # type: ignore
    grid["upper_surface"] = (tpms_scalar_field + surface_offset / 2.0).ravel(order="F")  # type: ignore

    tmp, upper_skeletal = grid.clip_scalar(
        scalars="upper_surface", invert=False, both=True
    )
    sheet, lower_skeletal = tmp.clip_scalar(scalars="lower_surface", both=True)
    del tmp
    sheet.clear_data()
    upper_skeletal.clear_data()
    lower_skeletal.clear_data()
    # TPMS and surface must be merged before due to periodic merge unstabilities
    # sheet.merge(sheet.extract_surface(), inplace=True)
    # lower_skeletal.merge(lower_skeletal.extract_surface(), inplace=True)
    # upper_skeletal.merge(upper_skeletal.extract_surface(), inplace=True)
    # tpms_parts = [
    #    _repeat_cell(tpms, repeat_cell)
    #    for tpms in (sheet, upper_skeletal, lower_skeletal)
    # ]
    return TriplyPeriodicStructure(sheet, upper_skeletal, lower_skeletal)


def generate_tpms(
    surface_function: SurfaceFunction,
    surface_offset: float,
    mesh_resolution: int,
    repeat_cell: Tuple[int, int, int] = (1, 1, 1),
    unit_cell_size: Tuple[float, float, float] = (1, 1, 1),
) -> TriplyPeriodicStructure:
    grid_x, grid_y, grid_z = _compute_mesh_grid(
        mesh_resolution=mesh_resolution,
        unit_cell_size=unit_cell_size,
        repeat_cell=repeat_cell,
    )
    grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
    k_x, k_y, k_z = np.divide(2 * np.pi, unit_cell_size)

    tpms_scalar_field = surface_function(k_x * grid_x, k_y * grid_y, k_z * grid_z)

    triply_periodic_minimal_surface_structure = _generate_tpms_parts(
        grid=grid,
        tpms_scalar_field=tpms_scalar_field,
        surface_offset=surface_offset,
        repeat_cell=repeat_cell,
    )
    return triply_periodic_minimal_surface_structure
