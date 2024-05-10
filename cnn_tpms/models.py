from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from cnn_tpms.typing import LabelDensity


class Linear1dBatchNormLReLUBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(inputs, outputs, bias=False),
            nn.BatchNorm1d(outputs),
            nn.LeakyReLU(0.2),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.linear_block(data)


class Convolution3dBatchNormLReLUBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
    ):
        super().__init__()
        self.convolution_block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.convolution_block(data)


class Convolution3dModel(nn.Module):
    def __init__(
        self,
        input_size: Tuple,
        in_channels: int = 1,
        n_classes: int = 9,
        features: Optional[List[int]] = None,
    ):
        super().__init__()
        if features is None:
            features = [16, 16, 32, 32, 64, 64]
        kernel_size = 3
        padding = 1
        layers: List[nn.Module] = []
        for i, feature in enumerate(features):
            layers.append(
                Convolution3dBatchNormLReLUBlock(
                    in_channels=in_channels,
                    out_channels=feature,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            if i % 2 != 0:
                layers.append(torch.nn.MaxPool3d(2))

            in_channels = feature
        layers.append(torch.nn.Flatten())
        num_maxpool_layers = len(features) // 2
        input_dense = features[-1] * np.prod(
            (np.asarray(input_size) // 2**num_maxpool_layers)
        )
        dense_units = [16, 16, 16]
        for units in dense_units:
            layers.append(
                Linear1dBatchNormLReLUBlock(inputs=input_dense, outputs=units)
            )
            input_dense = units
        layers.append(torch.nn.Linear(input_dense, n_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = self.layers(data)
        return data


class DensityClassifConvolutionModel(nn.Module):
    def __init__(
        self,
        convolution_model: Convolution3dModel,
    ) -> None:
        super().__init__()
        self.layers = convolution_model.layers
        self.classification_layer = self.layers[-1]
        self.layers = self.layers[:-1]  # Remove last layer
        self.regression_layer = nn.Sequential(
            nn.Linear(
                in_features=self.classification_layer.in_features,
                out_features=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, data: torch.Tensor) -> LabelDensity:
        features = self.layers(data)
        classification_output = self.classification_layer(features)
        regression_output = self.regression_layer(features)
        regression_output = regression_output.flatten()
        return LabelDensity(
            label=classification_output, density=regression_output
        )
