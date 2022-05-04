"""Main training script."""
import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

import models
import sigmaclast_data
import utils


def train_batch(net,
                dataloader,
                optimizer,
                objective,
                *,
                device=None,
                quiet=False,):
    """Standard pytorch training loop logic."""
    net.train()
    acc = total_loss = 0
    for X, y in tqdm(dataloader, leave=False, disable=True):
        if device:
            X, y = X.to(device), y.to(device)
        logits = net(X)
        loss = objective(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        acc += utils.accuracy_score_logits(logits, y, normalize=False)

    acc = acc / len(dataloader.dataset)

    return total_loss, acc


@torch.no_grad()
def validate(net,
             dataloader,
             objective,
             *,
             device=None,
             quiet=False,):
    """pytorch inference loop."""
    net.eval()
    acc = total_loss = 0.
    for X, y in tqdm(dataloader, leave=False, disable=quiet):
        if device:
            X, y = X.to(device), y.to(device)
        logits = net(X)
        acc += utils.accuracy_score_logits(logits, y, normalize=False)
        total_loss += objective(logits, y).item()

    acc = acc / len(dataloader.dataset)

    return total_loss, acc


def train(traindata, devdata, params):
    run = wandb.init(
        project="microstructures",
        config=params._asdict(),
        mode=params.wandb_mode
    )
    params = wandb.config

    logging.info("Training...")

    trainloader = DataLoader(
        traindata, batch_size=params.batch_size, shuffle=True)
    devloader  = DataLoader(
        devdata, batch_size=params.batch_size, shuffle=True)

    device = torch.device("cuda" if params.cuda else "cpu")

    # net = models.CNN(in_channels=3, img_sz=32, n_classes=2)
    net = models.VGG16(n_classes=2)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=params.lr)
    criterion = nn.CrossEntropyLoss()

    # utils.overfit_one_batch(net, devloader, optimizer, criterion)
    # exit()

    for epoch in tqdm(range(params.n_epochs), disable=params.quiet):
        trainloss, trainacc = train_batch(
            net, trainloader, optimizer, criterion,
            device=device, quiet=params.quiet
        )
        devloss, devacc = validate(
            net, devloader, criterion,
            device=device, quiet=params.quiet
        )

        wandb.log(
            {"train/loss": trainloss, "dev/loss": devloss,
             "train/acc": trainacc, "dev/acc": devacc},
            step=epoch
        )

        logging.info(f"{epoch=}")
        logging.info(f"\t{trainloss=:0.2f}, {trainacc=:0.2f}")
        logging.info(f"\t{devloss=:0.2f}, {devacc=:0.2f}")

    wandb.finish()


def cross_validate(config):
    data = sigmaclast_data.data_pipeline(config)
    validator = sigmaclast_data.CrossValidator(
        data,
        n_splits=config.n_splits,
        seed=config.seed,
    )

    for fold, (traindata, devdata) in enumerate(validator):
        logging.info(f"Cross validating...{fold=}")
        train(traindata, devdata, config)


def main(args, config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)

    if args.mode == "cv":
        cross_validate(config)
    else:
        data = sigmaclast_data.data_pipeline(config)
        devsize = round(config.dev_split * len(data))
        trainsize = len(data) - devsize
        traindata, devdata = random_split(data, [trainsize, devsize])
        train(traindata, devdata, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train microstructures clf.")
    parser.add_argument(
        "--mode", "-m", type=str, choices=("train", "cv"), default="train")
    args = parser.parse_args()

    config = utils.load_config()

    log_level = logging.WARNING if config.quiet else logging.INFO
    logging.basicConfig(level=log_level)

    main(args, config)
