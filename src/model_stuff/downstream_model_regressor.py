import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import copy
import wandb

from pytorch_lightning import LightningModule, Trainer
import torchvision.models as models
import torchmetrics

# FOR TESTING IF MY RESNET IS BETTER #
# from src.model_stuff.MyResNet import MyResNet

import re

class MyDownstreamModel(LightningModule):
    def __init__(self, backbone, max_epochs, lr=1e-4, num_classes=2, logger=None, dataloader_group_size=6, log_everything=False, freeze_backbone=True, fe="lightly", use_dropout=False, num_FC=2, use_LRa=False):
        super().__init__()
        self.num_classes=num_classes
        self.log_everything = log_everything
        self.lr = lr
        self.fe = fe # CAN BE lightly or myresnet
        self.use_dropout = use_dropout
        self.num_FC = num_FC
        self.use_LRa = use_LRa
        # self.save_hyperparameters()

        # just pass the feature extractor
        # print("\t backbone:", backbone)
        if fe == "lightly":
            self.feature_extractor = copy.deepcopy(backbone)
            print("----------------------- USING LIGHTLY ----------------------")
        elif fe == "myresnet":
            self.feature_extractor = MyResNet().backbone
            self.use_dropout = True
            print("----------------------- USING MYRESNET ----------------------")
        self.dataloader_group_size=dataloader_group_size
        self.parent_logger = logger
        # self.tensorboard_exp = self.parent_logger.experiment

        # freeze all parameters in feature extractor
        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        # trainable params
        in_dim = 512*self.dataloader_group_size
        # if self.use_dropout and self.num_FC==2:
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.6),
            torch.nn.Linear(in_dim, 512),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(512, 1),
            # torch.nn.Sigmoid(),
        )
        self.criteria = torch.nn.BCEWithLogitsLoss()

        # for logging
        self.downstream_val_losses = [] # avg per epoch
        self.downstream_val_accs = [] # avg per epoch
        self.downstream_train_losses = []
        self.downstream_train_accs = [] # avg per epoch

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
        # import pdb; pdb.set_trace()
        x = self.extract_features(x)
        x = torch.stack(torch.split(x, self.dataloader_group_size)).flatten(1) # bs x features
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        img_id, img_paths, y, x = batch
        x = x.view(x.shape[0]*x.shape[1], *x.shape[2:])
        # x = x.view(data_shape[0]*data_shape[1], data_shape[-1], *data_shape[2:4])
        
        out = self(x)

        loss = self.criteria(out.squeeze(-1), y.float())
        out_binarized = (torch.nn.functional.sigmoid(out) > 0.5).type(torch.uint8)
        acc = torchmetrics.functional.accuracy(out_binarized, y)

        if self.log_everything:
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            self.log('train_acc', acc, on_step=True, on_epoch=True)
        return {"loss": loss, "acc_downstream": acc, "batch_outputs": out.clone().detach()}

    def validation_step(self, batch, batch_idx):
        img_id, img_paths, y, x = batch
        x = x.view(x.shape[0]*x.shape[1], *x.shape[2:])

        out = self(x)

        val_loss = self.criteria(out.squeeze(-1), y.float())
        out_binarized = (torch.nn.functional.sigmoid(out) > 0.5).type(torch.uint8)
        val_acc = torchmetrics.functional.accuracy(out_binarized, y)

        if self.log_everything:
            self.log('val_loss', val_loss, on_step=True, on_epoch=True)
            self.log('val_acc', val_acc, on_step=True, on_epoch=True)

        return {"val_loss_downstream": val_loss.detach(), "val_acc_downstream": val_acc.detach(), "batch_outputs": out.clone().detach()}

    def get_preds(self, batch):
        img_id, img_paths, y, x = batch
        x = x.view(x.shape[0]*x.shape[1], *x.shape[2:])
        x = x.to(self.device)
        out = self(x)
        out = torch.nn.functional.sigmoid(out)
        return out


    def training_epoch_end(self, training_step_outputs):
        train_loss_downstream = []
        train_acc_downstream = []
        for dict_item in training_step_outputs:
            train_loss_downstream.append(dict_item["loss"].cpu().detach())
            train_acc_downstream.append(dict_item["acc_downstream"].cpu().detach())
        mean_train_loss = torch.mean(torch.Tensor(train_loss_downstream))
        mean_train_acc = torch.mean(torch.Tensor(train_acc_downstream))
        self.downstream_train_losses.append(float(mean_train_loss))
        self.downstream_train_accs.append(float(mean_train_acc))


    def validation_epoch_end(self, validation_step_outputs):
        # validation_step_outputs looks like a list with length=num_steps. Each index is a dict containing outputs for each step.
        
        # gather val_loss_downstream and val_acc_downstream
        val_loss_downstream = []
        val_acc_downstream = []
        for dict_item in validation_step_outputs:
            val_loss_downstream.append(dict_item["val_loss_downstream"].cpu().detach())
            val_acc_downstream.append(dict_item["val_acc_downstream"].cpu().detach())
        mean_val_loss = torch.mean(torch.Tensor(val_loss_downstream))
        mean_val_acc = torch.mean(torch.Tensor(val_acc_downstream))
        self.downstream_val_losses.append(float(mean_val_loss))
        self.downstream_val_accs.append(float(mean_val_acc))
        
    def configure_optimizers(self):
        if not self.use_LRa:
            optim = torch.optim.Adam(self.parameters(), self.lr)
            return optim
        else:
            optim = torch.optim.Adam(self.parameters(), self.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, )
            return [optim], [scheduler]
