import layers
import torch


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
        return layers.normalize_2nd_moment(latents)
