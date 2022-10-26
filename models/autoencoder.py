import torch

from models import layers


# ----------------------------------------------------------------------------
# Encdoer definitions
# ----------------------------------------------------------------------------
class Encoder(torch.nn.Module):
    def __init__(self, z_dim, img_resolution, img_channels, channel_base=128, norm_type="batch"):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        module_list = [
            layers.ConvLayer(
                in_channels=img_channels,
                out_channels=channel_base,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=False,
                activation="lrelu",
                norm_type="batch",
            )
        ]

        self.z_dim = z_dim
        cur_res = img_resolution // 2
        while cur_res > 4:
            block = layers.ConvLayer(
                channel_base,
                channel_base * 2,
                activation="lrelu",
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm_type=norm_type,
            )
            module_list.append(block)
            cur_res = cur_res // 2
            channel_base = channel_base * 2

        module_list.append(layers.Reshape(shape=(channel_base * cur_res**2,)))
        module_list.append(layers.LinearLayer(in_features=channel_base * cur_res**2, out_features=z_dim))
        self.encoder = torch.nn.Sequential(*module_list)

    def forward(self, img: torch.Tensor):
        latents = self.encoder(img)
        return latents


# ----------------------------------------------------------------------------
# Dcoder definitions
# ----------------------------------------------------------------------------
class Decoder(torch.nn.Module):
    def __init__(self, z_dim, img_resolution, img_channels, channel_base=1024, norm_type="batch"):
        super().__init__()

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.z_dim = z_dim
        cur_res = 4
        module_list = [
            layers.LinearLayer(
                in_features=z_dim,
                out_features=channel_base * cur_res**2,
                norm_type="batch",
                activation="lrelu",
            ),
            layers.Reshape(shape=(channel_base, cur_res, cur_res)),
        ]
        while cur_res < img_resolution:
            block = layers.ConvTranspsoeLayer(
                channel_base,
                channel_base // 2,
                activation="lrelu",
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                norm_type="batch",
            )
            module_list.append(block)
            cur_res = cur_res * 2
            channel_base = channel_base // 2

        module_list.append(
            layers.ConvTranspsoeLayer(
                channel_base,
                img_channels,
                activation="tanh",
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.encoder = torch.nn.Sequential(*module_list)

    def forward(self, img: torch.Tensor):
        return self.encoder(img)
