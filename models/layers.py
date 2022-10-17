"""
Basic layers used in the model definitions
"""

from typing import Callable

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# type definitions
# ----------------------------------------------------------------------------
ActivationFunction = Callable[[torch.Tensor], torch.Tensor]

# ----------------------------------------------------------------------------
# Helper functions and constants
# ----------------------------------------------------------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


def get_activation_function(activation: str) -> ActivationFunction:
    if activation == "relu":
        return torch.nn.ReLU()
    if activation == "lrelu":
        return torch.nn.LeakyReLU(0.2)
    if activation == "tanh":
        return torch.nn.Tanh()
    if activation == "sigmoid":
        return torch.nn.Sigmoid()
    if activation == "none":
        return torch.nn.Identity()


def get_norm_layer_2d(norm_type: str, num_features: int) -> torch.nn.Module:
    if norm_type == "batch":
        return torch.nn.BatchNorm2d(num_features)
    if norm_type == "instance":
        return torch.nn.InstanceNorm2d(num_features)
    if norm_type == "pixel":
        return PixelNormLayer()
    if norm_type == "none":
        return torch.nn.Identity()


def get_norm_layer_1d(norm_type: str, num_features: int) -> torch.nn.Module:
    if norm_type == "batch":
        return torch.nn.BatchNorm1d(num_features)
    if norm_type == "instance":
        return torch.nn.InstanceNorm1d(num_features)
    if norm_type == "pixel":
        return PixelNormLayer()
    if norm_type == "none":
        return torch.nn.Identity()


# ----------------------------------------------------------------------------
# utility classes
# ----------------------------------------------------------------------------
class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)


# ----------------------------------------------------------------------------
# normalization blocks
# ----------------------------------------------------------------------------
class PixelNormLayer(torch.nn.Module):
    """
    Pixelwise feature vector normalization.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


# ----------------------------------------------------------------------------
# upsampling and downsampling blocks
# ----------------------------------------------------------------------------
class UpSample(torch.nn.Module):
    def __init__(self, in_channels, factor=2, use_conv=True) -> None:
        super().__init__()
        self.resize = torch.nn.Upsample(scale_factor=factor, mode="nearest")
        self.conv = (
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) if use_conv else torch.nn.Identity()
        )

    def forward(self, x):
        return self.conv(self.resize(x))


class DownSample(torch.nn.Module):
    def __init__(self, in_channels, factor=2, use_conv=True) -> None:
        super().__init__()
        self.factor = factor
        self.use_conv = use_conv
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=factor, padding=1)

    def forward(self, x):
        if self.use_conv:
            # pad = (0, self.factor - 1, 0, self.factor - 1)
            # x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            return self.conv(x)
        else:
            return F.avg_pool2d(x, kernel_size=self.factor, stride=self.factor)


# ----------------------------------------------------------------------------
# fully connected block
# ----------------------------------------------------------------------------
class LinearLayer(torch.nn.Module):
    """
    Utility class for a linear layer followed by optional batch norm and activation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "none",
        norm_type: str = "none",
        bias: bool = True,
        use_sn: bool = False,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        if use_sn:
            self.linear = torch.nn.utils.spectral_norm(self.linear)

        self.act = get_activation_function(activation)
        self.norm_layer = get_norm_layer_1d(norm_type, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm_layer(x)
        return self.act(x)


# ----------------------------------------------------------------------------
# convolution block
# ----------------------------------------------------------------------------
class ConvLayer(torch.nn.Module):
    """
    Utility class for performing convolution followed by optional batch norm and activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "none",
        kernel_size: int = 3,
        stride: int = None,
        padding: int = None,
        bias: bool = True,
        norm_type: str = "batch",
        use_sn: bool = False,
    ):
        super().__init__()
        padding_ = int(kernel_size // 2) if padding is None else padding
        stride_ = 1 if stride is None else stride

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding_,
            stride=stride_,
            bias=bias,
        )

        if use_sn:
            self.conv = torch.nn.utils.spectral_norm(self.conv)

        self.act = get_activation_function(activation)
        self.norm_layer = get_norm_layer_2d(norm_type, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_layer(x)
        return self.act(x)


# ----------------------------------------------------------------------------
# transposed convolution block
# ----------------------------------------------------------------------------
class ConvTranspsoeLayer(torch.nn.Module):
    """
    Utility class for performing transposed convolution followed by optional, spectral normalization, batch normalization and activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = "none",
        stride: int = None,
        padding: int = None,
        output_padding: int = 0,
        bias: bool = True,
        norm_type: str = "none",
        use_sn: bool = False,
    ):
        super().__init__()
        padding_ = int(kernel_size // 2) if padding is None else padding
        stride_ = 1 if stride is None else stride

        self.conv_trans = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding_,
            stride=stride_,
            bias=bias,
            output_padding=output_padding,
        )

        if use_sn:
            self.conv_trans = torch.nn.utils.spectral_norm(self.conv_trans)

        self.act = get_activation_function(activation)
        self.norm_layer = get_norm_layer_2d(norm_type, out_channels)

    def forward(self, x):
        x = self.conv_trans(x)
        x = self.norm_layer(x)
        return self.act(x)
