"""Useful stand-alone functions."""
import yaml
from collections import namedtuple
from pathlib import Path
from typing import Callable, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def load_config(filename="config.yml"):
    """Load configuration values.

    Parameters
    ----------
    filename: pathlike
        Path to file.

    Returns
    -------
    namedtuple
    """
    with open(filename, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config["root"] = Path(config["root"])
    config = namedtuple("Params", config.keys())(**config)

    return config


def save_checkpoint(model_state: dict,
                    optim_state: dict,
                    filename: Union[str, Path],
                    **params) -> None:
    """Checkpoint model params during training."""
    checkpoint = {
        "model_state_dict": model_state,
        "optim_state_dict": optim_state
    }
    for key, val in params.items():
        checkpoint[key] = val

    torch.save(checkpoint, filename)


def load_checkpoint(filename: Union[str, Path]) -> dict:
    """Retrieve saved model state dict."""
    return torch.load(filename)


def accuracy_score_logits(logits: torch.Tensor,
                          true: torch.Tensor,
                          normalize: bool = False) -> Union[float, int]:
    """Calculate accuracy metric from logits tensor."""
    score = torch.sum(true == logits.argmax(dim=1)).item()

    return score / len(true) if normalize else score


def overfit_one_batch(model: nn.Module,
                      data: DataLoader,
                      optimizer: Optimizer,
                      objective: Callable,
                      n_epochs: int = 100,) -> None:
    model.train()
    X, y = next(iter(data))

    for i in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = objective(logits, y)
        if i % 10 == 0:
            print(f"{loss.item():0.3f}")
        loss.backward()
        optimizer.step()
