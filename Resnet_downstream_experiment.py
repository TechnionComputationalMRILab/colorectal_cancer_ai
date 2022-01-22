#!/usr/bin/env python
# coding: utf-8
# from src.data_stuff.pip_tools import install
# install(["pytorch-lightning", "seaborn", "timm"], quietly=True)
import torch
from pytorch_lightning import Trainer
# MY local imports
from src.callback_stuff import LogConfusionMatrix, PatientLevelValidation, DownstreamTrainer2
from src.data_stuff import tcga_datamodules
from src.model_stuff import MyResNet
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


print("---- RESNET Exp (tcga) W/Downstream ----")
print('CUDA available:', torch.cuda.is_available())

data_dir = "/home/shats/data/data/"
EXP_NAME = "tcga_Resnet_withDownstream"
logger = TensorBoardLogger("./lightning_logs", name=EXP_NAME)

model = MyResNet.MyResNet()
dm = tcga_datamodules.TcgaDataModule(data_dir=data_dir, batch_size=32)
class_to_idx = dm.get_class_to_idx_dict()

trainer = Trainer(gpus=1, max_epochs=3,
        logger=logger,
        move_metrics_to_cpu=True,
        # reload_dataloaders_every_epoch=True,
        # num_sanity_val_steps=0,
        callbacks=[
            # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
            # PatientLevelValidation.PatientLevelValidation(),
            DownstreamTrainer2.DownstreamTrainer(data_dir=data_dir)
            ],
        # precision=16
        )

trainer.fit(model, dm)

