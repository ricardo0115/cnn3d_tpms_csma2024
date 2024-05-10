from typing import Tuple

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torchinfo import summary

from cnn_tpms import models

DEFAULT_TEST_SEED = 69
MAX_EXAMPLES = 100
torch.manual_seed(DEFAULT_TEST_SEED)


@pytest.mark.parametrize(
    "model, random_data",
    [
        (models.Convolution3dModel, torch.rand((2, 1, 32, 32, 32))),
    ],
)
def test_classification_model_must_overfit_give_single_random_batch_data(
    model: torch.nn.Module, random_data: torch.Tensor
) -> None:
    # Arrange
    dummy_labels = torch.Tensor((0, 1)).type(torch.ByteTensor)
    model = model(input_size=random_data.shape[2:])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 5
    last_loss = torch.inf
    # Act and assert
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predicted = model(random_data)
        loss = criterion(predicted, dummy_labels)
        loss.backward()
        optimizer.step()
        assert loss < last_loss


@settings(max_examples=MAX_EXAMPLES)
@given(
    st.tuples(
        st.integers(min_value=20, max_value=64),
        st.integers(min_value=20, max_value=64),
        st.integers(min_value=20, max_value=64),
    )
)
def test_Convolution3dModel_given_arbitrary_input_size_must_compile_and_forward_data(
    arbitrary_input_size: Tuple[int, int],
) -> None:
    # Arrange
    model = models.Convolution3dModel(input_size=arbitrary_input_size)
    data = torch.rand((2, 1, *arbitrary_input_size))
    # Act and Assert
    assert model(data) is not None
