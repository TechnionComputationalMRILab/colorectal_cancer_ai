import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.data_stuff.patch_datamodule import TcgaDataModule
from src.data_stuff import tcga_moco_dm
# from src.model_stuff.MyResNet import MyResNet
from src.model_stuff.moco_model import MocoModel
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import argparse
from rich import print
# from src.data_stuff.pip_tools import install
# install(["pytorch-lightning", "albumentations", "seaborn", "timm", "wandb", "plotly", "lightly"], quietly=True)


if __name__ == "__main__":
    print(f"🚙 Starting Moco Experiment! 🚗")
    pl.seed_everything(42)

    # Data Dir and Model Checkpoint dir
    data_dir = "/workspace/repos/TCGA/data/"
    model_save_path = "/workspace/repos/hrdl/saved_models/moco/{EXP_NAME}"

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=610)
    args = parser.parse_args()

    # make experiment name
    EXP_NAME = f"MoCo_bs{args.batch_size}_ep{args.num_epochs}"
    print(f"\tExperiment Name: {EXP_NAME}")

    # logger
    logger=WandbLogger(project="colorectal_cancer_ai", name=EXP_NAME)

    # callbacks
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
            dirpath=model_save_path,
            filename='{epoch}-{MOCO_train_loss_ssl:.2f}',
            save_top_k=3,
            verbose=True,
            monitor='MOCO_train_loss_ssl',
            mode='min'
    )

    # model
    # need to pass max_epochs only for Cosine Learning rate annealing
    # embedder = MocoModel(hypers_dict["moco_max_epochs"])
    embedder = MocoModel()
    
    # data module
    dm = tcga_moco_dm.MocoDataModule(
            data_dir=hypers_dict["data_dir"],
            batch_size=hypers_dict["batch_size"], 
            subset_size=None,
            num_workers=os.cpu_count(),
            )
    
    trainer = Trainer(
            logger=logger,
            max_epochs=hypers_dict["moco_max_epochs"],
            callbacks=[lr_monitor_callback, checkpoint_callback,]
            )

    trainer.fit(embedder, dm)
