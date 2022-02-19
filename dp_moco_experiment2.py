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

import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# MY local imports
from src.callback_stuff import LogConfusionMatrix, PatientLevelValidation, DownstreamTrainer2
from src.data_stuff import tcga_datamodules, tcga_moco_dm
from src.model_stuff import moco_model
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processing args...")
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--strat', type=str, default="ddp")
    # for batch size, 128 works for 4nodes, 4gpus(total)

    args = parser.parse_args()

    # --- hypers --- #
    data_subset=None
    batch_size=args.batch_size
    num_nodes=args.num_nodes
    gpus=args.gpus
    num_workers=16
    strat=args.strat
    # ------------- #

    EXP_NAME = f"MOCO_{ON_SERVER}_datasz{data_subset}_bs{batch_size}_gpus{gpus}_nodes{num_nodes}_strat{strat}"
    logger = WandbLogger(project="moti", name=EXP_NAME)

    print("---- MOCO Experiment (tcga) ----")
    print(f"experiment name: {EXP_NAME}")
    print('CUDA available:', torch.cuda.is_available())

    memory_bank_size = 4096
    moco_max_epochs = 600
    # downstream_max_epochs = 60
    # downstream_test_every = 50
    model = moco_model.MocoModel(memory_bank_size, moco_max_epochs)
    dm = tcga_moco_dm.MocoDataModule(data_dir=data_dir,
            batch_size=batch_size, 
            subset_size=data_subset,
            num_workers=num_workers)


    # monitors
    checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir+"moco",
        filename='{epoch}-{MOCO_train_loss_ssl:.2f}',
        save_top_k=3,
        verbose=True,
        monitor='MOCO_train_loss_ssl',
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(gpus=gpus,
            num_nodes=num_nodes,
            strategy=strat,
            max_epochs=moco_max_epochs,
            logger=logger,
            callbacks=[
                # LogConfusionMatrix.LogConfusionMatrix(class_to_idx),
                # DownstreamTrainer2.DownstreamTrainer(data_dir=data_dir,
                #     downstream_max_epochs=20,
                #     downstream_subset_size=data_subset,
                #     do_first_n_epochs=2,
                #     do_every_n_epochs=200,
                #     downstream_batch_size=batch_size,
                #     do_on_train_epoch_end=True),
                checkpoint_callback,
                lr_monitor
                ],
            )
    trainer.fit(model, dm)

