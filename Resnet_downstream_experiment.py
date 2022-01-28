#!/usr/bin/env python
# coding: utf-8
# from src.data_stuff.pip_tools import install
# install(["pytorch-lightning", "seaborn", "timm", "wandb", "plotly"], quietly=True)
import torch
from pytorch_lightning import Trainer
# MY local imports
from src.callback_stuff import LogConfusionMatrix, PatientLevelValidation, DownstreamTrainer2
from src.data_stuff import tcga_datamodules
from src.model_stuff import MyResNet
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# ON_SERVER = "DGX"
ON_SERVER = "moti"

print("---- RESNET Exp (tcga) W/Downstream ----")
print('CUDA available:', torch.cuda.is_available())

data_dir=None
if ON_SERVER == "DGX":
    data_dir="/workspace/repos/TCGA/data/"
elif ON_SERVER == "moti":
    data_dir="/home/shats/data/data/"

EXP_NAME = f"tcga_Resnet_withDownstream_{ON_SERVER}_downstream0.7"
logger = WandbLogger(project="moti", name=EXP_NAME)
# logger = TensorBoardLogger("./lightning_logs", name=EXP_NAME)

model = MyResNet.MyResNet()
dm = tcga_datamodules.TcgaDataModule(data_dir=data_dir, batch_size=64, fast_subset=False)
class_to_idx = dm.get_class_to_idx_dict()

trainer = Trainer(gpus=1, max_epochs=40,
        logger=logger,
        move_metrics_to_cpu=True,
        # reload_dataloaders_every_epoch=True,
        # num_sanity_val_steps=0,
        callbacks=[
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            # PatientLevelValidation.PatientLevelValidation(),
            DownstreamTrainer2.DownstreamTrainer(data_dir=data_dir,
                downstream_max_epochs=15,
                downstream_subset_size=0.62,
                do_every_n_epochs=3,
                downstream_batch_size=64)
            ],
        # precision=16
        )

trainer.fit(model, dm)

