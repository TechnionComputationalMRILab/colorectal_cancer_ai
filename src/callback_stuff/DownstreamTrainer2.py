import torch
import torchvision
import pytorch_lightning as pl
from collections import defaultdict
import tempfile
from itertools import zip_longest
import os
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly
import plotly.express as px
import plotly.graph_objects as go

# local imports
from ..model_stuff.DownstreamModel import MyDownstreamModel
from ..data_stuff.data_tools import get_patient_name_from_path
from src.data_stuff.downstream_dataset import DownstreamTrainingDataset

class DownstreamTrainer(pl.Callback):
    def __init__(self, data_dir="/workspace/repos/TCGA/data/",
            downstream_max_epochs=5,
            downstream_lr=1e-5,
            downstream_group_sz=4,
            downstream_subset_size=None,
            do_first_n_epochs=0,
            do_every_n_epochs=2,
            downstream_batch_size=32,
            do_on_train_epoch_end=False) -> None:
        print("Downsteam Evaluation initialized")
        """
        Need to build a dict of {patient: [p1e, p2e, ... pne]} where pne is the embedding for patch #n of the patient
        Since n changes, I will just take the first (or random) j embeddings
        Build a picture (matrix) of these embeddings and train on a patient level this way.
        """

        self.data_dir = data_dir
        self.downstream_max_epochs = downstream_max_epochs
        self.downstream_lr = downstream_lr
        self.downstream_subset_size = downstream_subset_size
        self.do_first_n_epochs = do_first_n_epochs
        self.do_every_n_epochs = do_every_n_epochs
        self.downstream_batch_size = downstream_batch_size
        self.do_on_train_epoch_end=do_on_train_epoch_end

        # where we will store training metrics
        self.val_acc_record_df = pd.DataFrame()
        self.val_loss_record_df = pd.DataFrame()
        self.train_acc_record_df = pd.DataFrame()
        self.train_loss_record_df = pd.DataFrame()

        # dataframe passed to model to record stuff
        # self.metric_record_df = pd.DataFrame(columns=['epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss'])
        # self.metric_record_df = pd.DataFrame()
        # NOTE: this has been moved to be created inside the downstream training model

        # initialize wandb table (used to plot training scores over time)
        # need as many columns as we have epochs for this run
        # self.downstream_val_accs_table = wandb.Table(columns = [str(i) for i in list(range(self.downstream_max_epochs+1))])
        # self.downstream_val_accs_table = 

        # initialize downstream dataset/dataloader
        rgb_mean = (0.4914, 0.4822, 0.4465)
        rgb_std = (0.2023, 0.1994, 0.2010)
        train_transforms = torchvision.transforms.Compose([
            # torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(rgb_mean, rgb_std),
        ])
        val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(rgb_mean, rgb_std),
        ])

        self.training_group_size = 8

        self.train_ds = DownstreamTrainingDataset(root_dir=data_dir, transform=train_transforms, dataset_type="train", group_size=self.training_group_size, subset_size=self.downstream_subset_size)
        self.val_ds = DownstreamTrainingDataset(root_dir=data_dir, transform=val_transforms, dataset_type="test", group_size=self.training_group_size, subset_size=self.downstream_subset_size)
        self.train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=self.downstream_batch_size, shuffle=True, num_workers=8)
        self.val_dl = torch.utils.data.DataLoader(self.val_ds, batch_size=self.downstream_batch_size, shuffle=False, num_workers=8)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.do_on_train_epoch_end:
            print("doing epoch end...")
            self.on_validation_epoch_end(trainer, pl_module)


    def on_validation_epoch_end(self, trainer, pl_module):
        
        # log for the first 3 epochs and then after that only log when required:
        if trainer.current_epoch < self.do_first_n_epochs or trainer.current_epoch%self.do_every_n_epochs==0:

            # make dataloader from original data that loads it like this:
            print("\n\n---- IN DOWNSTREAM TRAINER callback ----")
            backbone = pl_module.feature_extractor.backbone
            logger = pl_module.logger
            model = MyDownstreamModel(backbone=backbone,
                    num_classes=2,
                    logger=logger,
                    dataloader_group_size=self.training_group_size,
                    )
            
            # train model and log highest acc after 
            # self.train_downstream_model(trainer, model, logger, self.train_dl, self.val_dl)
            downstream_trainer = pl.Trainer(gpus=1, max_epochs=self.downstream_max_epochs, logger=logger, move_metrics_to_cpu=True)
            downstream_trainer.fit(model, self.train_dl, self.val_dl)

            # done training. record metrics
            epoch_val_accs = model.downstream_val_accs # list eg [0.5, 0.6, 0.75]
            epoch_val_losses = model.downstream_val_losses
            epoch_train_accs = model.downstream_train_accs
            epoch_train_losses = model.downstream_train_losses

            # append these metrics to what we already have in the df based on moco epoch num
            moco_epoch = trainer.current_epoch
            self.val_acc_record_df[f"ep_{moco_epoch}"] = epoch_val_accs
            self.val_loss_record_df[f"ep_{moco_epoch}"] = epoch_val_accs
            self.train_acc_record_df[f"ep_{moco_epoch}"] = epoch_val_accs
            self.train_loss_record_df[f"ep_{moco_epoch}"] = epoch_val_accs

            # plot
            self.wandb_log_metric_df(self.val_acc_record_df, name="Downstream Val Accuracy")
            self.wandb_log_metric_df(self.val_loss_record_df, name="Downstream Val Loss")
            self.wandb_log_metric_df(self.train_acc_record_df, name="Downstream Train Accuracy")
            self.wandb_log_metric_df(self.train_loss_record_df, name="Downstream Train Loss")



    def wandb_log_metric_df(self, df, name="Downstream Val Accuracy"):
        fig = go.Figure()
        custom_colors_list = self.make_plotly_custom_colors_list(len(df.columns))
        fig.layout.colorway = custom_colors_list
        for idx, col_name in enumerate(df):
            fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col_name],
                        mode='lines+markers',
                        name=col_name,
                        ),
                    )
        wandb.log({name: fig})


    
    def make_plotly_custom_colors_list(self, num_colors):
        # making a custom plotly colors list
        """
        EX:
        ['rgb(0, 0, 0)',
         'rgb(0, 0, 15)',
         'rgb(0, 0, 31)',
         'rgb(0, 0, 47)',
         'rgb(0, 0, 63)',
         'rgb(0, 0, 79)',
         ...
         ]
        """
        custom_colors_list = []
        for i in range(num_colors):
            custom_colors_list.append(f"rgb(0, 0, {int((255/num_colors)*i)})")
        return custom_colors_list
