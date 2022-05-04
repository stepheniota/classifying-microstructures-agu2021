"""Dataclasses to work with Sigmaclast images."""
from pathlib import Path
from typing import Union
import itertools

import cv2
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


DATAPATH = Path("../data/processed_data")


class SigmaclastFolder(ImageFolder):
    """pytorch-friendly dataclass to work with sigmaclast imgs."""
    def find_classes(self, directory):
        labels = ("CW", "CCW")
        cls2idx = {label: i for i, label in enumerate(labels)}

        return labels, cls2idx


class CrossValidator:
    """Performs kfold validation on torch Datasets.

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        Dataset to cross validate.
    n_splits: int
        Number of validation splits desired.
    seed: int
        Random seed. Currently unused.

    Attributes
    ----------
    cv: sklearn.model_selection.KFold
        Generator returning train and dev indices for each fold.
    n_samples: int
        Length of full dataset.
    idx: np.array
        Valid dataset indices.
    """
    def __init__(self, dataset, n_splits, seed=None):
        self.dataset = dataset
        self.n_splits = n_splits
        self.n_samples = len(dataset)
        self.idx = np.arange(self.n_samples)
        kfold = KFold(n_splits=n_splits,) # random_state=seed,)
        self.cv = kfold.split(self.idx)

    def __iter__(self):
        for _ in range(self.n_splits):
            train_idx, dev_idx = next(self.cv)
            traindataset = Subset(self.dataset, train_idx)
            devdataset = Subset(self.dataset, dev_idx)

            yield traindataset, devdataset


def preprocess_dataset(root, quiet=False):
    r"""Flips sigmaclast dataset by pi radians.

    A sigmaclast's rotation is anti-symetric. By rotating a
    sample by pi radians, its label flips from `CW` to `CCW`
    or vice-versa. Thus, this preprocessing procedure effectively
    doubles the dataset.

    Assumes data are arranged in this way by default:
    ```
        root/original_data/{CW, CCW}/xxx.{png, jpg}
        root/original_data/{CW, CCW}/xxy.{png, jpg}
        ...
    ```
    Outputs preprocessed data in `root/processed_data/{CW, CCW}/`.

    Parameters
    ----------
    root: Path or str
        Path to parent directory where imgs are stored.
        See directory assumptions in docstring.
    quiet: bool = True
        Quiet output to stdout.
    """
    root = Path(root)
    original_dir = root/"original_data"
    CW_dir = root/"processed_data/CW"
    CCW_dir = root/"processed_data/CCW"
    CW_dir.mkdir(parents=True, exist_ok=True)
    CCW_dir.mkdir(parents=True, exist_ok=True)
    img_count = itertools.count()

    def preprocess_img(img_file, label):
        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        # `1` flips img horizonally.
        # This flips the label as well.
        # TODO: add explanation why...
        img_flipped = cv2.flip(img, 1)
        img_num = next(img_count)
        name = f"img_{img_num}_{label}.jpg"
        label_flipped = "CCW" if label == "CW" else "CW"
        name_flipped = f"img_{img_num}_{label_flipped}.jpg"

        try:
            if label == "CW":
                cv2.imwrite(str(CW_dir/name), img)
                cv2.imwrite(str(CCW_dir/name_flipped), img_flipped)
            else:
                cv2.imwrite(str(CCW_dir/name), img)
                cv2.imwrite(str(CW_dir/name_flipped), img_flipped)
        except cv2.error as e:
            if not quiet:
                print(f"Error: {e}"); print("Continuing...")

    for directory in original_dir.iterdir():
        label = directory.parts[-1]
        if label == "CW" or label == "CCW":
            for img_file in directory.iterdir():
                preprocess_img(img_file, label)

    return None


def data_pipeline(config):
    """Pipeline that abstracts datapreprocessing steps."""
    if config.build_data:
        preprocess_dataset(config.root)

    transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=config.MEAN, std=config.STD),
                    T.Resize((32, 32))
                ])

    dataset = SigmaclastFolder(
        root=config.root/"processed_data", transform=transform)

    return dataset
