from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import einops
import numpy as np
import pandas as pd
import pyvista as pv
import torch
from torch.utils.data import Dataset

from cnn_tpms import voxel
from cnn_tpms.typing import LabelDensity


def rotation_transform(shape: pv.PolyData) -> pv.PolyData:
    vector_rotation = np.random.uniform(-1, 1, size=3)
    angle = np.random.uniform(0, 360)
    shape = shape.rotate_vector(
        vector=vector_rotation,
        angle=angle,
        inplace=True,
        point=np.mean(shape.points, axis=0),
    )
    return shape


@dataclass
class LatticeStlVolumes(Dataset):
    dataframe: pd.DataFrame
    voxel_resolution: int
    volume_grid_shape: Tuple[int, int, int]
    transform: Optional[Callable] = None

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, LabelDensity]:
        filename = self.dataframe.iloc[index, 0]
        density = self.dataframe.iloc[index, 1]
        density = np.float32(density)
        label = self.dataframe.iloc[index, 3]
        shape = pv.wrap(np.load(filename)["points"])
        if self.transform:  # Rotation
            shape = self.transform(shape)
        volume = torch.from_numpy(
            voxel.voxelize(
                shape.points, self.voxel_resolution, self.volume_grid_shape
            )
        ).type(torch.FloatTensor)
        volume = einops.rearrange(volume, "d w h -> 1 d w h")
        return volume, LabelDensity(label=label, density=density)
