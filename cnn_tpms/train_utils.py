from typing import List, Optional, Tuple

import torch

Metrics = Tuple[List[float], List[float]]


def save_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }
    if optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: torch.nn.Module,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return epoch
