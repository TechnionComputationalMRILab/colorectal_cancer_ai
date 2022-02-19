#!/usr/bin/env python
# coding: utf-8
from src.data_stuff.pip_tools import install
install(["pytorch-lightning", "seaborn", "timm"], quietly=True)
import torch
from pytorch_lightning import Trainer
# MY local imports
from src.callback_stuff import LogConfusionMatrix, PatientLevelValidation, DownstreamTrainer
from src.data_stuff import tcga_datamodules, imagenette_datamodules
from src.model_stuff import MyResNet
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


print("---- RESNET Exp (imagenette) ----")
print('CUDA available:', torch.cuda.is_available())

EXP_NAME = "imagenette_Resnet_HIRES"
logger = TensorBoardLogger("/workspace/repos/lightning_logs", name=EXP_NAME)

# dm = tcga_datamodules.TcgaDataModule(batch_size=128)
dm = imagenette_datamodules.ImagenetteDataModule(batch_size=4)
model = MyResNet.MyResNet(num_classes=len(dm.get_class_to_idx_dict()))
class_to_idx = dm.get_class_to_idx_dict()

trainer = Trainer(gpus=1, max_epochs=45,
        logger=logger,
        # reload_dataloaders_every_epoch=True,
        # callbacks=[
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            # PatientLevelValidation.PatientLevelValidation(),
            # DownstreamTrainer.DownstreamTrainer()
            # ],
        )

trainer.fit(model, dm)

