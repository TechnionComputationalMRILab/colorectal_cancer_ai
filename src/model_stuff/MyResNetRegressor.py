import numpy as np
import pandas as pd
import torch
import torchvision

from pytorch_lightning import LightningModule, Trainer
import torchvision.models as models
import torchmetrics

class MyResNet(LightningModule):
    def __init__(self, num_classes=2, weight_decay=None):
        super().__init__()
        self.save_hyperparameters()
        self.weight_decay = weight_decay

        resnet = models.resnet18(pretrained=True)
        self.backbone = torch.nn.Sequential(*(list(resnet.children())[:-1])) # just remove the fc
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 1),
            # torch.nn.Sigmoid(),
        )
        self.criteria = torch.nn.BCEWithLogitsLoss()

    def extract_features(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1) # (batch_sz, 512, 1, 1) -> (batch_sz, 512)
        return x
        
    def forward(self, x):
        x = self.extract_features(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        img_id, img_paths, y, x = batch

        # important note
        # x.shape = torch.Size([bs, 1, 3, 224, 224])
        # I can squeeze the 2nd dim or reshape. reshape is more general
        x = x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        out = self(x)

        loss = self.criteria(out.squeeze(-1), y.float())
        out_binarized = (torch.nn.functional.sigmoid(out) > 0.5).type(torch.uint8)
        acc = torchmetrics.functional.accuracy(out_binarized, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True)
        loss = loss.unsqueeze(dim=-1)
        return {"loss": loss, "acc": acc, "batch_outputs": out.clone().detach()}

    def validation_step(self, batch, batch_idx):
        img_id, img_paths, y, x = batch

        # important note
        # x.shape = torch.Size([bs, 1, 3, 224, 224])
        # I can squeeze the 2nd dim or reshape. reshape is more general
        x = x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        out = self(x)
        
        # import pdb; pdb.set_trace()
        val_loss = self.criteria(out.squeeze(-1), y.float())
        out_binarized = (torch.nn.functional.sigmoid(out) > 0.5).type(torch.uint8)
        val_acc = torchmetrics.functional.accuracy(out_binarized, y)
        
        self.log('val_loss', val_loss, on_step=True, on_epoch=True)
        self.log('val_acc', val_acc, on_step=True, on_epoch=True)
        val_loss = val_loss.unsqueeze(dim=-1)
        return {"val_loss": val_loss, "val_acc": val_acc, "batch_outputs": out.clone().detach()}

    def get_preds(self, batch):
        img_id, img_paths, y, x = batch
        x = x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        x = x.to(self.device)
        out = self(x)
        out = torch.nn.functional.sigmoid(out)
        return out

    # on end of train/validation, I can print stuff like this:
    # print(f"REGULAR train loss: {train_loss} | train acc: {train_acc}")
    # print(f"REGULAR val loss: {val_loss} | val acc: {val_acc}")
                
    def configure_optimizers(self):
        if self.weight_decay:
            print(f"DOING WEIGHT DECAY {self.weight_decay}")
            return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)
        else:
            print(" DOING LEARNING RATE 1e-4!!!")
            return torch.optim.Adam(self.parameters(), lr=1e-4)
