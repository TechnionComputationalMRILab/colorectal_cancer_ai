import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt

from pytorch_lightning import LightningModule, Trainer
import torchvision.models as models
import torchmetrics
import timm

from ..data_stuff import dataset_tools
import re

class MySwinTransformer(LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model("swin_tiny_patch4_window7_224", num_classes=0, pretrained=True, in_chans=3)
        num_features = self.model.num_features
        self.fc = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(num_features, self.hparams.num_classes), torch.nn.Sigmoid())
        # self.criteria = torch.nn.BCEWithLogitsLoss()
        self.criteria = torch.nn.BCELoss()

        
    def forward(self, x):
        f = self.model(x)
        out = self.fc(f)
        return out
        
    def training_step(self, batch, batch_idx):
        path, x, y = batch
        out = self(x)

        loss = self.criteria(out, torch.nn.functional.one_hot(y, self.hparams.num_classes).float())
        acc = torchmetrics.functional.accuracy(torch.argmax(out, dim=1), y)
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        loss = loss.unsqueeze(dim=-1)
        return {"loss": loss, "acc": acc, "batch_outputs": out}

    def validation_step(self, batch, batch_idx):
        path, x, y = batch
        out = self(x)

        
        val_loss = self.criteria(out, torch.nn.functional.one_hot(y, self.hparams.num_classes).float())
        val_acc = torchmetrics.functional.accuracy(torch.argmax(out, dim=1), y)
        
        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc)
        val_loss = val_loss.unsqueeze(dim=-1)
        return {"val_loss": val_loss, "val_acc": val_acc, "batch_outputs": out}
    
                
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)
    
