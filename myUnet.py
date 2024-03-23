import torch
from torch import nn
from torch.utils.data import Subset, Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from typing import Optional, Union, List
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class myUnet(SegmentationModel):

    def __init__(self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == 'cuda': torch.backends.cudnn.benchmark = False

        self.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, amsgrad=True)
        self.scaler = None
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=1)


    def fit(self, train_dl: DataLoader,  val_dl: DataLoader, num_epochs, metric, stat_period=2):

        train_epoch_loss, val_epoch_loss = [], []
        train_epoch_metric, val_epoch_metric = [], []

        for epoch in range(num_epochs):
            epoch_loss = {"train": 0, "val": 0}
            epoch_metr = {"train": 0, "val": 0}

            self.train()
            with tqdm(train_dl, desc=f"Epoch [{epoch}] train process") as tepoch:
                for train_batch in tepoch:

                    self.optimizer.zero_grad()
                    img = train_batch[0].to(self.device)
                    mask = train_batch[1].to(self.device, dtype=torch.long)

                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            out_model = self(img)
                            model_losses = self.criterion(out_model, mask)

                        self.scaler.scale(model_losses).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        out_model = self(img)
                        model_losses = self.criterion(out_model, mask)
                        out_model = out_model.cpu()
                        mask = mask.cpu()
                        mm = metric(out_model.argmax(axis=1).numpy(), mask.numpy())
                        model_losses.backward()
                        self.optimizer.step()

                    tepoch.set_postfix(loss=model_losses.item(), jaccard_index=mm)
                    if not (epoch % stat_period):
                        epoch_loss['train'] += model_losses.item()
                        epoch_metr['train'] += np.sum(mm)/3

            if not (epoch % stat_period):
                self.eval()
                with torch.no_grad():
                    for val_batch in val_dl:
                        img = val_batch[0].to(self.device)
                        mask = val_batch[1].to(self.device, dtype=torch.long)

                        out_model = self(img)
                        model_losses = self.criterion(out_model, mask)
                        epoch_loss['val'] += model_losses.item()
                        epoch_metr['val'] += np.sum(mm)/3

                train_epoch_loss.append(epoch_loss["train"] / len(train_dl))
                val_epoch_loss.append(epoch_loss["val"] / len(val_dl))

                train_epoch_metric.append(epoch_metr["train"] / len(train_dl))
                val_epoch_metric.append(epoch_metr["val"] / len(val_dl))

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(epoch_metr['val'])


    def predict(self, data):
        return self(data)


    def save(self, name="unet.pt"):
        torch.save(self, name)
