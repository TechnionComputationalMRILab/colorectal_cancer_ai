import torch
import torchvision
import pytorch_lightning as pl
from collections import defaultdict
import tempfile
from itertools import zip_longest
import os
from tqdm import tqdm

# local imports
from ..model_stuff.DownstreamModel import MyDownstreamModel
from ..data_stuff.data_tools import get_patient_name_from_path
from src.data_stuff.downstream_dataset import DownstreamTrainingDataset

class DownstreamTrainer(pl.Callback):
    def __init__(self) -> None:
        print("Downsteam Evaluation initialized")
        """
        Need to build a dict of {patient: [p1e, p2e, ... pne]} where pne is the embedding for patch #n of the patient
        Since n changes, I will just take the first (or random) j embeddings
        Build a picture (matrix) of these embeddings and train on a patient level this way.
        """
        
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

        self.train_ds = DownstreamTrainingDataset(root_dir="/workspace/repos/TCGA/data/", transform=train_transforms, dataset_type="train")
        self.val_ds = DownstreamTrainingDataset(root_dir="/workspace/repos/TCGA/data/", transform=val_transforms, dataset_type="test")
        self.train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=2, shuffle=True, num_workers=4)
        self.val_dl = torch.utils.data.DataLoader(self.val_ds, batch_size=2, shuffle=True, num_workers=4)

    def train_downstream_model(self, model, logger, train_dl, val_dl):
        for sample in tqdm(train_dl):
            # sample.keys() = dict_keys(['label', 'patient_id', 'data_paths', 'data'])
            y = list(zip(*sample["label"])) # unzip the nonsense that the pytorch dataloader did
            d_shape = sample["data"].shape
            nwhc_data = sample["data"].view(d_shape[0]*d_shape[1], *d_shape[2:]).shape
            import pdb; pdb.set_trace()
            y_hat = model(nwhc_data)


    def on_validation_epoch_end(self, trainer, pl_module):

        # make dataloader from original data that loads it like this:
        print("---- IN DOWNSTREAM TRAINER callback ----")
        backbone = pl_module.feature_extractor
        logger = pl_module.logger
        model = MyDownstreamModel(backbone=backbone, num_classes=2, logger=logger)
        
        # train model and log highest acc after 
        self.train_downstream_model(model, logger, self.train_dl, self.val_dl)


