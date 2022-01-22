#!/usr/bin/env python
# coding: utf-8
# from src.data_stuff.pip_tools import install
# install(["pytorch-lightning", "seaborn", "timm"], quietly=True)
import torch
from pytorch_lightning import Trainer
# MY local imports
from src.callback_stuff import LogConfusionMatrix, PatientLevelValidation 
from src.data_stuff import tcga_datamodules
from src.model_stuff import MyResNet
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


print("---- RESNET Exp (tcga) ----")
print('CUDA available:', torch.cuda.is_available())

EXP_NAME = "tcga_Resnet"
logger = TensorBoardLogger("./lightning_logs", name=EXP_NAME)

model = MyResNet.MyResNet()
dm = tcga_datamodules.TcgaDataModule(data_dir="/home/shats/data/data/", batch_size=32)
class_to_idx = dm.get_class_to_idx_dict()

trainer = Trainer(gpus=1, max_epochs=6,
        logger=logger,
        callbacks=[
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            # PatientLevelValidation.PatientLevelValidation(),  # ])
            ],
        )

trainer.fit(model, dm)

