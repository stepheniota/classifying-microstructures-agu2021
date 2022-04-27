"""Dataclasses to work with Sigmaclast images."""
from pathlib import Path
from typing import Union
import itertools

import cv2
from torchvision.datasets import ImageFolder


def preprocess_dataset(root: Union[Path, str], quiet: bool = False) -> None:
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
            if quiet is False:
                print(f"Error: {e}")
                print(f"Continuing...")

    for directory in original_dir.iterdir():
        label = directory.parts[-1]
        if label == "CW" or label == "CCW":
            for img_file in directory.iterdir():
                preprocess_img(img_file, label)

    return None


class SigmaclastFolder(ImageFolder):
    """pytorch-friendly dataclass to work with sigmaclast imgs."""
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        labels = ("CW", "CCW")
        cls2idx = {label: i for i, label in enumerate(labels)}

        return labels, cls2idx
