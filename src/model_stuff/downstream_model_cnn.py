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
from src.model_stuff.MyResNet import MyResNet

import re

class MyDownstreamModel(LightningModule):
    def __init__(self, backbone, max_epochs, lr=1e-4, num_classes=2, logger=None, dataloader_group_size=6, log_everything=False, freeze_backbone=True):
        super().__init__()
        self.num_classes=num_classes
        self.log_everything = log_everything
        self.lr = lr
        # self.fe = fe # CAN BE lightly or myresnet
        # self.use_dropout = use_dropout
        # self.num_FC = num_FC
        # self.use_LRa = use_LRa
        # self.save_hyperparameters()

        # just pass the feature extractor
        # print("\t backbone:", backbone)
        self.feature_extractor = copy.deepcopy(backbone)
        self.dataloader_group_size=dataloader_group_size
        self.parent_logger = logger
        # self.tensorboard_exp = self.parent_logger.experiment

        # freeze all parameters in feature extractor
        if freeze_backbone:
            print("... freezing moco backbone ...")
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        # trainable params
        in_dim = 512
        num_channels = self.dataloader_group_size
        self.c_blk = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=5, stride=2),
                # torch.nn.BatchNorm1d(num_channels*2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(in_channels=num_channels*2, out_channels=num_channels, kernel_size=3, stride=2),
                # torch.nn.BatchNorm1d(num_channels),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2),
                )
        if self.dataloader_group_size == 4:
            self.fc = torch.nn.Sequential(
                    torch.nn.Linear(252, 252),
                    torch.nn.Linear(252, self.num_classes),
                    )
        elif self.dataloader_group_size == 8:
            self.fc = torch.nn.Sequential(
                    # torch.nn.Linear(504, 504),
                    torch.nn.Linear(504, self.num_classes),
                    )
        elif self.dataloader_group_size == 16:
            self.fc = torch.nn.Sequential(
                    # torch.nn.Linear(1008, 1008),
                    torch.nn.Linear(1008, self.num_classes),
                    )
        elif self.dataloader_group_size == 32:
            self.fc = torch.nn.Sequential(
                    # torch.nn.Linear(2016, 256),
                    torch.nn.Linear(2016, self.num_classes),
                    )
        # elif self.dataloader_group_size == 2:
        #     self.fc = torch.nn.Sequential(
        #             torch.nn.Linear(126, 126),
        #             torch.nn.Linear(126, self.num_classes),
        #             )
        # elif self.dataloader_group_size == 3:
        #     self.fc = torch.nn.Sequential(
        #             torch.nn.Linear(189, 189),
        #             torch.nn.Linear(189, self.num_classes),
        #             )
        # in_dim = 512*self.dataloader_group_size
        # if self.use_dropout and self.num_FC==2:
        #     self.fc = torch.nn.Sequential(
        #         torch.nn.Dropout(0.6),
        #         torch.nn.Linear(in_dim, 512),
        #         torch.nn.Dropout(0.6),
        #         torch.nn.Linear(512, self.num_classes),
        #         # torch.nn.Sigmoid(),
        #     )
        # elif (not self.use_dropout and self.num_FC==2):
        #     self.fc = torch.nn.Sequential(
        #         # torch.nn.Dropout(0.6),
        #         torch.nn.Linear(in_dim, 512),
        #         # torch.nn.Dropout(0.6),
        #         torch.nn.Linear(512, self.num_classes),
        #         # torch.nn.Sigmoid(),
        #     )
        # elif self.num_FC==1:
        #     self.fc = torch.nn.Sequential(
        #         torch.nn.Linear(in_dim, self.num_classes),
        #     )


        


        self.criteria = torch.nn.BCEWithLogitsLoss()
        # self.criteria = torch.nn.BCELoss()

        # for logging
        # self.downstream_val_accs_table = downstream_val_accs_table
        # self.full_run_train_acc_record = full_run_train_acc_record
        self.downstream_val_losses = [] # avg per epoch
        self.downstream_val_accs = [] # avg per epoch
        self.downstream_train_losses = []
        self.downstream_train_accs = [] # avg per epoch


    # overload logging so that it logs with the logger used for the 
    # trained feature extractor
    # def log(name, value, on_step, on_epoch):
    #    return self.parent_logger.log(name, value, on_step=on_step, on_epoch=on_epoch)

    def extract_features(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1) # (batch_sz, 512, 1, 1) -> (batch_sz, 512)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        
        # x = torch.stack(torch.split(x, self.dataloader_group_size)).flatten(1) # bs x features
        x = torch.stack(torch.split(x, self.dataloader_group_size)) # bs x gs x features
        x = self.c_blk(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        img_id, img_paths, y, x = batch
        x = x.view(x.shape[0]*x.shape[1], *x.shape[2:])
        # x = x.view(data_shape[0]*data_shape[1], data_shape[-1], *data_shape[2:4])
        
        out = self(x)

        loss = self.criteria(out, torch.nn.functional.one_hot(y, self.num_classes).float())
        acc = torchmetrics.functional.accuracy(torch.argmax(out, dim=1), y)
        loss = loss.unsqueeze(dim=-1)

        # , "batch_outputs_downstream": out.clone().detach()}
        if self.log_everything:
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            self.log('train_acc', acc, on_step=True, on_epoch=True)
        return {"loss": loss, "acc_downstream": acc, "batch_outputs": out.clone().detach()}

    def validation_step(self, batch, batch_idx):
        img_id, img_paths, y, x = batch
        x = x.view(x.shape[0]*x.shape[1], *x.shape[2:])

        out = self(x)

        val_loss = self.criteria(out, torch.nn.functional.one_hot(y.long(), self.num_classes).float())
        val_acc = torchmetrics.functional.accuracy(torch.argmax(out, dim=1), y)
        if self.log_everything:
            self.log('val_loss', val_loss, on_step=True, on_epoch=True)
            self.log('val_acc', val_acc, on_step=True, on_epoch=True)
        val_loss = val_loss.unsqueeze(dim=-1)
        del batch

        #, "batch_outputs_downstream": out.clone().detach()}
        return {"val_loss_downstream": val_loss.detach(), "val_acc_downstream": val_acc.detach(), "batch_outputs": out.clone().detach()}

    def get_preds(self, batch):
        img_id, img_paths, y, x = batch
        x = x.view(x.shape[0]*x.shape[1], *x.shape[2:])
        x = x.to(self.device)
        out = self(x)
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
        optim = torch.optim.Adam(self.parameters(), self.lr)
        return optim
