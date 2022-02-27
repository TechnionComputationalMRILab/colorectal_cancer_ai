#!/usr/bin/env python
# coding: utf-8

# ON_SERVER = "DGX"
# ON_SERVER = "moti"
ON_SERVER = "Alsx2"
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
from pytorch_lightning import Trainer
# MY local imports
from src.callback_stuff import LogConfusionMatrix, PatientLevelValidation 
from src.data_stuff import tcga_datamodules
from src.model_stuff import MyResNet
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

print("---- RESNET Exp (tcga) ----")
print('CUDA available:', torch.cuda.is_available())

# --- hypers --- #
min_patches_per_patient = 8
# ------------- #

EXP_NAME = f"tcga_Resnet_minpatch{min_patches_per_patient}"
logger=WandbLogger(project="moti", name=EXP_NAME)
# logger = TensorBoardLogger("./lightning_logs", name=EXP_NAME)

model = MyResNet.MyResNet()
dm = tcga_datamodules.TcgaDataModule(data_dir=data_dir, batch_size=64, fast_subset=False, min_patches_per_patient=min_patches_per_patient)
class_to_idx = dm.get_class_to_idx_dict()

trainer = Trainer(gpus=1, max_epochs=35,
        logger=logger,
        callbacks=[
            PatientLevelValidation.PatientLevelValidation(),
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            ]
        )

trainer.fit(model, dm)

