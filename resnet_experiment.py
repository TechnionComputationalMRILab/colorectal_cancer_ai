import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import argparse

from src.data_stuff.NEW_patch_dataset import PatchDataModule
from src.model_stuff.MyResNet import MyResNet
from src.callback_stuff.PatientLevelValidation import PatientLevelValidation
# from src.data_stuff.pip_tools import install
# install(["pytorch-lightning", "albumentations", "seaborn", "timm", "wandb", "plotly", "lightly"], quietly=True)


if __name__ == "__main__":
    print(f"🚙 Starting Resnet Experiment! 🚗")
    pl.seed_everything(42)

    # Data Dir and Model Checkpoint dir
    data_dir = "/workspace/repos/data/tcga_data_formatted/"
    model_save_path = "/workspace/repos/hrdl/saved_models/moco/{EXP_NAME}"

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()

    # make experiment name
    EXP_NAME = f"Resnet_baseline_bs{bs}"
    print(f"\tExperiment Name: {EXP_NAME}")

    # logger
    logger=WandbLogger(project="colorectal_cancer_ai", name=EXP_NAME)

    # callbacks
    patient_level_validation_callback = PatientLevelValidation(group_size=args.group_size, debug_mode=False)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename='{epoch}-{val_majority_vote_acc:.2f}-{val_acc_epoch}',
        save_top_k=3,
        verbose=True,
        monitor='val_majority_vote_acc',
        mode='max'
    )

    # model
    model = MyResNet()

    # data
    dm = PatchDataModule(
            data_dir=args.data_dir, 
            batch_size=args.batch_size, 
            group_size=args.group_size,
            num_workers=os.cpu_count(),
            )

    # trainer
    trainer = Trainer(
            logger=logger,
            callbacks=[checkpoint_callback, patient_level_validation_callback]
            )

    trainer.fit(model, dm)
