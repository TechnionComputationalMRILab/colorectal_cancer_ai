#!/usr/bin/env python
# coding: utf-8
# from src.data_stuff.pip_tools import install
# install(["pytorch-lightning", "seaborn", "timm", "wandb", "plotly"], quietly=True) #opencv-python too
import torch
from pytorch_lightning import Trainer
# MY local imports
from src.callback_stuff import LogConfusionMatrix, PatientLevelValidation, DownstreamTrainer2
from src.data_stuff import tcga_datamodules, tcga_moco_dm
from src.model_stuff import moco_model
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# ON_SERVER = "DGX"
# ON_SERVER = "moti"
ON_SERVER = "Alsx2"

print("---- MOCO Experiment (tcga) ----")
print('CUDA available:', torch.cuda.is_available())

data_dir=None
if ON_SERVER == "DGX":
    data_dir="/workspace/repos/TCGA/data/"
elif ON_SERVER == "moti":
    data_dir="/home/shats/data/data/"
elif ON_SERVER == "Alsx2":
    data_dir="/home/shats/data/data/"

EXP_NAME = f"tcga_MOCO_{ON_SERVER}_"
logger = WandbLogger(project="moti", name=EXP_NAME)

memory_bank_size = 4096
moco_max_epochs = 900
# downstream_max_epochs = 60
# downstream_test_every = 50
model = moco_model.MocoModel(memory_bank_size, moco_max_epochs)
dm = tcga_moco_dm.MocoDataModule(data_dir=data_dir, fast_subset=True)

trainer = Trainer(gpus=1, max_epochs=moco_max_epochs,
        logger=logger,
        callbacks=[
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            DownstreamTrainer2.DownstreamTrainer(data_dir=data_dir,
                downstream_max_epochs=18,
                downstream_subset_size=None,
                start_downstream_epoch=3,
                do_every_n_epochs=5,
                downstream_batch_size=64,
                do_on_train_epoch_end=True)
            ],
        )
trainer.fit(model, dm)

