#!/usr/bin/env python
# coding: utf-8
from src.data_stuff.pip_tools import install
install(["pytorch-lightning", "seaborn", "timm"], quietly=True)
import torch
from pytorch_lightning import Trainer
# MY local imports
from src.callback_stuff import LogConfusionMatrix, PatientLevelValidation, DownstreamTrainer
from src.data_stuff import tcga_datamodules
from src.model_stuff import MyResNet
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


print("---- RESNET Exp (tcga) ----")
print('CUDA available:', torch.cuda.is_available())

EXP_NAME = "tcga_Resnet"
logger = TensorBoardLogger("/workspace/repos/lightning_logs", name=EXP_NAME)

model = MyResNet.MyResNet()
dm = tcga_datamodules.TcgaDataModule(batch_size=128)
class_to_idx = dm.class_to_idx

trainer = Trainer(gpus=1, max_epochs=3,
                  logger=logger,
                  callbacks=[
    LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
    PatientLevelValidation.PatientLevelValidation(),  # ])
    DownstreamTrainer.DownstreamTrainer()])

trainer.fit(model, dm)

