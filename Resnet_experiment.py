#!/usr/bin/env python
# coding: utf-8
from src.data_stuff.pip_tools import install
install(["pytorch-lightning", "seaborn", "timm", "wandb"], quietly=True)
import torch
from pytorch_lightning import Trainer
# MY local imports
from src.callback_stuff import LogConfusionMatrix, PatientLevelValidation 
from src.data_stuff import tcga_datamodules
from src.model_stuff import MyResNet
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

ON_SERVER = "DGX"
# ON_SERVER = "moti"

print("---- RESNET Exp (tcga) ----")
print('CUDA available:', torch.cuda.is_available())

data_dir=None
if ON_SERVER == "DGX":
    data_dir="/workspace/repos/TCGA/data/"
elif ON_SERVER == "moti":
    data_dir="/home/shats/data/data/"

EXP_NAME = "tcga_Resnet"
logger=WandbLogger(project="moti", name=EXP_NAME)
# logger = TensorBoardLogger("./lightning_logs", name=EXP_NAME)

model = MyResNet.MyResNet()
dm = tcga_datamodules.TcgaDataModule(data_dir=data_dir, batch_size=64, fast_subset=False)
class_to_idx = dm.get_class_to_idx_dict()

trainer = Trainer(gpus=1, max_epochs=35,
        logger=logger,
        callbacks=[
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            DownstreamTrainer2.DownstreamTrainer(data_dir=data_dir,
                downstream_max_epochs=18,
                downstream_subset_size=None,
                start_downstream_epoch=3,
                do_every_n_epochs=5,
                downstream_batch_size=64)
            ],
        )

trainer.fit(model, dm)

