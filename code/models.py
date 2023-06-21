"""Neural network models."""
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as M


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 with MLP classifier for finetuning.

    Parameters
    ----------
    n_classes: int
        Number of classes.
    hidden: sequence[int] (optional)
        The hidden dimensions of the MLP classifier.
        Defaults to (256, 128).
    act: nn.Module (optional)
        Activation function for MLP classifer.
        Defaults to ReLU if none specified.

    Attributes
    ----------
    model: InceptionV3
        Pretrained model with no learnable parameters.
    clf: MLP
        Custom classification head. Learnable.
    """
    def __init__(self, n_classes=2, hidden=None, act=None):
        super().__init__()
        model = M.inception_v3(pretrained=True)
        model.requires_grad_(False)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        self.model = model

        self.clf = MLP(in_features, *hidden, n_classes, act=act)

    def forward(self, x):
        """Returns logits."""
        with torch.no_grad():
            features = self.model(x)
        logits = self.clf(features)


class VGG19(nn.Module):
    """Pretrained VGG19 with MLP classifier for finetuning.

    Parameters
    ----------
    n_classes: int
        Number of classes.
    hidden: sequence[int] (optional)
        The hidden dimensions of the MLP classifier.
        Defaults to (256, 128).
    act: nn.Module (optional)
        Activation function for MLP classifer.
        Defaults to ReLU if none specified.

    Attributes
    ----------
    model: VGG19
        Pretrained model with no learnable parameters.
    clf: MLP
        Custom classification head. Learnable.
    """
    def __init__(self, n_classes=2, hidden=None, act=None):
        super().__init__()
        model = M.vgg19(pretrained=True)
        model.requires_grad_(False)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Identity()
        self.model = model

        self.clf = MLP(in_features, *hidden, n_classes, act=act)

    def forward(self, x):
        """Returns logits."""
        with torch.no_grad():
            features = self.model(x)
        logits = self.clf(features)

        return logits


class ResNet50(nn.Module):
    """Pretrained ResNet18 with MLP classifier for finetuning.

    Parameters
    ----------
    n_classes: int
        Number of classes.
    hidden: sequence[int] (optional)
        The hidden dimensions of the MLP classifier.
        Defaults to (256, 128).
    act: nn.Module (optional)
        Activation function for MLP classifer.
        Defaults to ReLU if none specified.

    Attributes
    ----------
    model: ResNet50
        Pretrained model with no learnable parameters.
    clf: MLP
        Custom classification head. Learnable.
    """

    def __init__(self, n_classes=2, hidden=None, act=None):
        super().__init__()
        model = M.resnet50(pretrained=True)
        model.requires_grad_(False)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        self.model = model

        self.clf = MLP(in_features, hidden, n_classes, act=act)

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
        logits = self.clf(features)

        return logits


class ResNet18(nn.Module):
    """Pretrained ResNet18 with MLP classifier for finetuning.

    Parameters
    ----------
    n_classes: int
        Number of classes.
    hidden: sequence[int] (optional)
        The hidden dimensions of the MLP classifier.
        Defaults to (256, 128).
    act: nn.Module (optional)
        Activation function for MLP classifer.
        Defaults to ReLU if none specified.

    Attributes
    ----------
    model: ResNet18
        Pretrained model with no learnable parameters.
    clf: MLP
        Custom classification head. Learnable.
    """

    def __init__(self, n_classes=2, hidden=None, act=None):
        super().__init__()
        model = M.resnet18(pretrained=True)
        model.requires_grad_(False)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        self.model = model

        self.clf = MLP(in_features, hidden, n_classes, act=act)

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
        logits = self.clf(features)

        return logits


class CNN(nn.Module):
    """Simple CNN classifier.

    Assumes input tensor of shape
        ``(batch_sz, in_channels, img_sz, img_sz)``.

    Parameters
    ----------
    in_channels: int
        Number of input channels of each img (e.g., 3 for rgb imgs).
    img_sz: int
        Width/height of imput imgs. Assumes imgs are square.
    n_classes: int
        Number of target classes.

    Attributes
    ----------
    cnn: nn.Sequential
        Convolutional layers.
    classifier: nn.Sequential
        FFNN that returns logits.
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
    """Simple Multilayer Perceptron.

    Parameters
    ----------
    *dims: sequence of ints
        Dimensions of hidden layers.
    act: optional[nn.Module]
        Activation function. Default=nn.ReLU

    Attributes
    ----------
    layers: nn.ModuleList
        Learnable modules of the model.
    act: nn.Module
        Activation function.
    """
    def __init__(self, *dims, act=None):
        super().__init__()
        print(dims)

        self.act = act if act else nn.ReLU()
        self.layers = nn.ModuleList([
            nn.Linear(dims[-1], dims[i]) for i in range(1, len(dims))
        ])

    def forward(self, x):
        """Returns logits."""
        for i, m in enumerate(self.layers):
            x = m(x)
            if len(self.layers) - 1 != i:
                x = self.act(x)

        return x


if __name__ == "__main__":
    # Manual tests
    batch = 16
    in_features = 32
    n_classes = 2
    hidden = [128, 256, 128]
    m = MLP(hidden, act=None)
    print(list(m.modules()))

    inp = torch.rand(batch, in_features)
    out = m(inp)

    print(out.shape)
