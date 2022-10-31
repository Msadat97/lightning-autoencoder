from turtle import forward

import pytorch_lightning as pl
import torch


class AutoEncoderTask(pl.LightningModule):
    def __init__(self, encoder, decoder, loss_fn, opt_kwargs, scheduler_kwargs=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.opt_kwargs = opt_kwargs
        self.loss = loss_fn
        self.scheduler_kwargs = scheduler_kwargs

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.Adam(params, **self.opt_kwargs)
        if self.scheduler_kwargs is not None:
            scheduler = None
            # scheduler = torch.optim.lr_scheduler(opt, **self.scheduler_kwargs)
            return [opt], [scheduler]
        return opt

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
