"""Useful stand-alone functions."""
import argparse
from pathlib import Path
from typing import Callable, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def save_checkpoint(
    model_state: dict,
    optim_state: dict,
    file_name: Union[str, Path],
    **params
) -> None:
    """Checkpoint model params during training."""
    checkpoint = {
        "model_state_dict": model_state,
        "optim_state_dict": optim_state
    }
    for key, val in params.items():
        checkpoint[key] = val

    torch.save(checkpoint, file_name)


def load_checkpoint(file_name: Union[str, Path]) -> dict:
    """Retrieve saved model state dict."""
    return torch.load(file_name)


def accuracy_score_logits(
    logits: torch.tensor,
    true: torch.tensor,
    normalize: bool = False
) -> Union[float, int]:
    score = torch.sum(true == logits.argmax(dim=1)).item()

    return score / len(true) if normalize else score


def overfit_one_batch(
    model: nn.Module,
    data: DataLoader,
    optimizer: Optimizer,
    objective: Callable,
    n_epochs: int = 100,
) -> None:
    model.train()
    X, y = next(iter(data))

    for i in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = objective(logits, y)
        if i % 10 == 0:
            print(f"{loss.item():0.4f}")
        loss.backward()
        optimizer.step()


"""
def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Train model for microstructure classification task.")

    parser.add_argument("--batch_size", type=int, help="Training batch size.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--dev_split", type=float,
        help="Fraction of dataset to hold out for cross-validation.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--cuda", type=bool, help="Access to GPU?")
    parser.add_argument("--model", type=str, help="Which model to train.")
    parser.add_argument("--logdir", type=str, help="Where to save logs.")
    parser.add_argument("--n_epochs", type=int, help="How many training epochs.")
    parser.add_argument("--quiet", type=bool, help="Silence output to stdout?")

    args = parser.parse_args()

    return dict(vars(args))
"""
