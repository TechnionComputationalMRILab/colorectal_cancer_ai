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
            do_every_n_epochs=2,
            downstream_batch_size=32) -> None:
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
        self.do_every_n_epochs = do_every_n_epochs
        self.downstream_batch_size = downstream_batch_size

        # full_run_train/val_acc/loss_record is a list of lists.
        # each list contains the training losses/accs for that run
        # so len(full_run_val_acc_record) == number of times I train a downstream trainer
        # and each list in it eg len(full_run_val_acc_record[0]) == downstream_max_epochs
        self.full_run_val_acc_record = [] 
        self.full_run_train_acc_record = []

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

    def on_validation_epoch_end(self, trainer, pl_module):
        
        # log for the first 3 epochs and then after that only log when required:
        if trainer.current_epoch < 3 or trainer.current_epoch%self.do_every_n_epochs==0:

            # make dataloader from original data that loads it like this:
            print("\n\n---- IN DOWNSTREAM TRAINER callback ----")
            backbone = pl_module.feature_extractor
            logger = pl_module.logger
            model = MyDownstreamModel(backbone=backbone, num_classes=2, logger=logger, dataloader_group_size=self.training_group_size,
                    full_run_val_acc_record=self.full_run_val_acc_record,
                    full_run_train_acc_record=self.full_run_train_acc_record)
            
            # train model and log highest acc after 
            # self.train_downstream_model(trainer, model, logger, self.train_dl, self.val_dl)
            downstream_trainer = pl.Trainer(gpus=1, max_epochs=self.downstream_max_epochs, logger=logger, move_metrics_to_cpu=True)
            downstream_trainer.fit(model, self.train_dl, self.val_dl)
            
            num_lines = trainer.max_epochs
            colors = np.linspace(0, 1, num_lines+1)
            
            # if trainer.current_epoch == trainer.max_epochs:
            plt.figure()
            for ep, accs_list in enumerate(self.full_run_val_acc_record):
                plt.plot(accs_list, c=(colors[ep], 0, 0), label=f'ep_{ep}')
                plt.ylabel("accuracy")
                plt.xlabel("downstream epoch")
                plt.legend()
            wandb.log({"Downstream Val Accuracy": plt})

            plt.figure()
            for ep, accs_list in enumerate(self.full_run_train_acc_record):
                plt.plot(accs_list, c=(colors[ep], 0, 0), label=f'ep_{ep}')
                plt.ylabel("accuracy")
                plt.xlabel("downstream epoch")
                plt.legend()
            wandb.log({"Downstream Train Accuracy": plt})

