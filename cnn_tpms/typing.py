from __future__ import annotations

from typing import NamedTuple, Union

import torch


class LabelDensity(NamedTuple):
    label: torch.Tensor
    density: torch.Tensor

    def to(self, device: Union[str, int]) -> LabelDensity:
        label = self.label.to(device)
        density = self.density.to(device)
        return LabelDensity(label, density)
