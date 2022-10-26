from turtle import forward

import pytorch_lightning as pl
import torch


class AutoEncoderTask(pl.LightningModule):
    def __init__(self, encoder, decoder, loss_fn, configs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.configs = configs
        self.loss = loss_fn

    def configure_optimizers(self):
        return super().configure_optimizers()

    def forward(self, x):
        return self.encoder(x)

    def reconstruct(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.reconstruct(x)
        train_loss = self.loss(x, x_hat)
        self.log("train_loss", train_loss)
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.reconstruct(x)
        val_loss = self.loss(x, x_hat)
        self.log("val_loss", val_loss)
        return {"val_loss": val_loss}
