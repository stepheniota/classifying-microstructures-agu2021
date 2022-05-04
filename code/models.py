"""Models."""
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    """Simple CNN classifier.

    Assumes input tensor of shape
        ``(batch_sz, in_channels, img_sz, img_sz)``.
    """
    def __init__(
        self,
        in_channels=3,
        img_sz=32,
        n_classes=2,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # (batch, 6, img_sz, img_sz)
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2
            ),  # (batch, 6, img_sz / 2, img_sz / 2)
            nn.Conv2d(
                6, 12, 5, 1, 2),  # (batch, 12, img_sz / 2, img_sz)
            nn.ReLU(),
            nn.MaxPool2d(2)  # (batch, 12, img_sz / 4, img_sz / 4)
        )
        img_sz = img_sz / 4
        self.classifier = nn.Sequential(
            nn.Linear(int(12 * np.square(img_sz)), 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        """Fwd pass, returns logits."""
        hidden = self.cnn(x)
        hidden_ = torch.flatten(hidden, start_dim=1, end_dim=-1)
        logits = self.classifier(hidden_)

        return logits


class MLP(nn.Module):
    """Simple FFNN."""
    def __init__(self, in_features, n_classes=2):
        super().__init__()
        self.features = nn.Linear(in_features, 128)
        self.dense = nn.Linear(128, 256)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        """Returns logits."""
        features = torch.relu(
            self.features(x)
        )
        dense = torch.relu(
            self.dense(features)
        )
        logits = self.classifier(dense)

        return logits


if __name__ == "__main__":
    # Manual tests
    batch = 32
    in_channels = 3
    img_sz = 32
    n_classes = 2

    net = CNN(
        in_channels=in_channels, img_sz=img_sz, n_classes=n_classes,
    )
    inp = torch.rand(batch, in_channels, img_sz, img_sz)
    print(inp.shape)

    out = net(inp)
    print(out.shape)

    in_features = 64
    inp = torch.rand(batch, in_features)
    net = MLP(in_features=in_features, n_classes=n_classes)

    out = net(inp)
    print(out.shape)
