from typing import Optional, Tuple

import numpy as np


def voxelize(
    points: np.ndarray,
    voxel_resolution: int,
    grid_shape: Optional[Tuple[int, int, int]] = None,
    # grid shape must be > than voxel_resolution
) -> np.ndarray:
    # Representative points
    normalized_points = (
        (points - points.min()) / (points.max() - points.min())
    ) - 0.5
    representative_indices = (
        np.unique(normalized_points - np.min(normalized_points, axis=0), axis=0)
        * (voxel_resolution - 1)
    ).astype(np.uint32)
    if not grid_shape:  # If not grid_shape use representative boundaries
        grid_shape = tuple(np.max(representative_indices, axis=0) + 1)
    volume = np.zeros(grid_shape, dtype=bool)
    volume[tuple(representative_indices.T)] = True
    return volume
