#!/usr/bin/env python
# coding: utf-8

# pip install pytorch-lightning seaborn timm wandb plotly lightly opencv-python

ON_SERVER = "DGX"
# ON_SERVER = "moti"
# ON_SERVER = "Alsx2"
# ON_SERVER = "argus"

data_dir=None
checkpoint_dir = "./saved_models/"
if ON_SERVER == "DGX":
    data_dir="/workspace/repos/TCGA/data/"
    checkpoint_dir = "/workspace/repos/colorectal_cancer_ai/saved_models/"
    from src.data_stuff.pip_tools import install
    install(["pytorch-lightning", "seaborn", "timm", "wandb", "plotly", "lightly"], quietly=True)
elif ON_SERVER == "argus":
    data_dir="/tcmldrive/databases/Public/TCGA/data/"
elif ON_SERVER == "moti":
    data_dir="/home/shats/data/data/"
elif ON_SERVER == "Alsx2":
    data_dir="/home/shats/data/data/"

import torch
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# MY local imports
from src.callback_stuff import LogConfusionMatrix, PatientLevelValidation, DownstreamTrainer2
from src.data_stuff import tcga_datamodules, tcga_moco_dm
from src.model_stuff import moco_model
from src.model_stuff.DownstreamModel import MyDownstreamModel
from src.data_stuff.data_tools import get_patient_name_from_path
from src.data_stuff.downstream_dataset import DownstreamTrainingDataset
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


# --- hypers --- #
data_subset=None
batch_size=64
# num_nodes=1
gpus=1
num_workers=16
# strat="ddp"
training_group_size = 12
# ------------- #

EXP_NAME = f"tcga_mocoDOWNSTREAM_{ON_SERVER}_datasz{data_subset}_bs{batch_size}_gpus{gpus}_tgs{training_group_size}"
logger = WandbLogger(project="moti", name=EXP_NAME)

print("---- MOCO Experiment (tcga) ----")
print(f"experiment name: {EXP_NAME}")
print('CUDA available:', torch.cuda.is_available())

### ---- DATA STUFF ---- ###
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


train_ds = DownstreamTrainingDataset(
        root_dir=data_dir, 
        transform=train_transforms, 
        dataset_type="train", 
        group_size=training_group_size, 
        subset_size=data_subset)
val_ds = DownstreamTrainingDataset(
        root_dir=data_dir, 
        transform=val_transforms, 
        dataset_type="test", 
        group_size=training_group_size, 
        subset_size=data_subset)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)

### ---- MODEL STUFF ---- ###
memory_bank_size = 4096
moco_max_epochs = 900
# read model
# model_loc = "/workspace/repos/colorectal_cancer_ai/saved_models_from_elsewhere/epoch=183-MOCO_train_loss_ssl=1.65.ckpt"
model_loc = "/workspace/repos/colorectal_cancer_ai/saved_models/moco/epoch=500-MOCO_train_loss_ssl=0.89.ckpt"
model = moco_model.MocoModel(memory_bank_size, moco_max_epochs).load_from_checkpoint(model_loc, memory_bank_size=memory_bank_size)

backbone = model.feature_extractor.backbone
model = MyDownstreamModel(backbone=backbone, num_classes=2, logger=logger, dataloader_group_size=training_group_size, log_everything=True)


trainer = Trainer(gpus=gpus,
        max_epochs=80,
        logger=logger,
        )
trainer.fit(model, train_dl, val_dl)
