import pytorch_lightning as pl
import lightly
import torch.nn as nn
import torch
import numpy as np
import copy

class MocoModel(pl.LightningModule):
    def __init__(self, memory_bank_size, moco_max_epochs=None):
        super().__init__()
        
        # need for cosine annealing
        self.moco_max_epochs = moco_max_epochs

        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', 1, num_splits=8)
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco based on ResNet
        self.num_ftrs = 512
        self.feature_extractor =\
            lightly.models.MoCo(backbone, num_ftrs=self.num_ftrs, m=0.99, batch_shuffle=True)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def forward(self, x):
        # self.resnet_moco(x)
        self.feature_extractor(x)

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        # y0, y1 = self.resnet_moco(x0, x1)
        y0, y1 = self.feature_extractor(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('MOCO_train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.feature_extractor.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.moco_max_epochs)
        return [optim], [scheduler]


