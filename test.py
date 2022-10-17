from torchsummary import summary

from models import autoencoder, layers

decoder = autoencoder.Decoder(
    z_dim=128, img_resolution=28, img_channels=3, channel_base=1024, norm_type="batch"
).cuda()
summary(decoder, (128,))
