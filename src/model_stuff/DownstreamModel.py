import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import copy

from pytorch_lightning import LightningModule, Trainer
import torchvision.models as models
import torchmetrics

from ..data_stuff import dataset_tools
import re

class MyDownstreamModel(LightningModule):
    def __init__(self, backbone, num_classes=2, logger=None, dataloader_group_size=6):
        super().__init__()
        self.save_hyperparameters()

        # just pass the feature extractor
        # print("\t backbone:", backbone)
        self.feature_extractor = copy.deepcopy(backbone)
        self.dataloader_group_size=dataloader_group_size

        # freeze all parameters in feature extractor
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # trainable params
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2560, 1024),
            torch.nn.Linear(1024, self.hparams.num_classes),
            torch.nn.Sigmoid(),
        )
        # self.criteria = torch.nn.BCEWithLogitsLoss()
        self.criteria = torch.nn.BCELoss()

    # overload logging so that it logs with the logger used for the 
    # trained feature extractor
    def log(name, value, on_step, on_epoch):
        return self.parent_logger(name, value, on_step, on_epoch)

    def extract_features(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1) # (batch_sz, 512, 1, 1) -> (batch_sz, 512)
        return x

    def forward(self, x):
        # WARNING: this is hacky
        # so x is a vector like this [10x3x224x224]
        # but 10 is actually batch_size*group_size
        # so in this case, group_size is 5 and batch size is 2
        # so I need to split this into 2 after extracting features.
        # Then I can concat them and run through the linear head.
        # so need to split after extracting features 
        x = self.extract_features(x)
        x = torch.stack(torch.split(x, self.dataloader_group_size)).flatten(1) # bs x features
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        y = batch["label"]
        # y = list(zip(*batch["label"]))
        # y = torch.Tensor([item for sublist in y for item in sublist]).long().to(self.device) # flatten y from list of tuples
        patient_ids = batch["patient_id"]
        data_paths = batch["data_paths"]
        data_shape = batch["data"].shape
        x = batch["data"].view(data_shape[0]*data_shape[1], data_shape[-1], *data_shape[2:4])
        
        out = self(x)

        loss = self.criteria(out, torch.nn.functional.one_hot(y, self.hparams.num_classes).float())
        acc = torchmetrics.functional.accuracy(torch.argmax(out, dim=1), y)
        
        # self.log('train_loss', loss, on_step=True, on_epoch=True)
        # self.log('train_acc', acc, on_step=True, on_epoch=True)
        # if self.current_epoch == self.max_epochs:
        #     print("\t✅ Logging last epoch train loss/acc")
        #     self.log('downstream_train_loss', loss, on_step=False, on_epoch=True)
        #     self.log('downstream_train_acc', acc, on_step=False, on_epoch=True)
        loss = loss.unsqueeze(dim=-1)
        return {"loss": loss, "acc_downstream": acc, "batch_outputs_downstream": out.clone().detach()}

    def validation_step(self, batch, batch_idx):
        y = batch["label"]
        # y = list(zip(*batch["label"])) # list of tuples
        # y = torch.Tensor([item for sublist in y for item in sublist]).long().to(self.device) # flatten y from list of tuples
        patient_ids = batch["patient_id"]
        data_paths = batch["data_paths"]
        data_shape = batch["data"].shape
        x = batch["data"].view(data_shape[0]*data_shape[1], data_shape[-1], *data_shape[2:4])

        out = self(x)

        val_loss = self.criteria(out, torch.nn.functional.one_hot(y.long(), self.hparams.num_classes).float())
        val_acc = torchmetrics.functional.accuracy(torch.argmax(out, dim=1), y)
        
        # self.log('val_loss', val_loss, on_step=True, on_epoch=True)
        # self.log('val_acc', val_acc, on_step=True, on_epoch=True)
        # if self.current_epoch == self.max_epochs:
        #     print("\t✅ Logging last epoch val loss/acc")
        #     self.log('downstream_val_loss', val_loss, on_step=False, on_epoch=True)
        #     self.log('downstream_val_acc', val_acc, on_step=False, on_epoch=True)
        val_loss = val_loss.unsqueeze(dim=-1)
        return {"val_loss_downstream": val_loss, "val_acc_downstream": val_acc, "batch_outputs_downstream": out.clone().detach()}

    def validation_epoch_end(self, validation_step_outputs):
        # validation_step_outputs looks like a list with length=num_steps. Each index is a dict containing outputs for each step.
        
        # gather val_loss_downstream and val_acc_downstream
        val_loss_downstream = []
        val_acc_downstream = []
        for dict_item in validation_step_outputs:
            val_loss_downstream.append(dict_item["val_loss_downstream"].cpu().detach())
            val_acc_downstream.append(dict_item["val_acc_downstream"].cpu().detach())
        mean_loss = torch.mean(torch.Tensor(val_loss_downstream))
        mean_acc = torch.mean(torch.Tensor(val_acc_downstream))

        print(f"\t --- Downstream mean loss: {mean_loss} ---")
        print(f"\t --- Downstream mean acc:  {mean_acc} ---")
    
                
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
