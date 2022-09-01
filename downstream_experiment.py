import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torchvision
import argparse
from src.model_stuff.moco_model import MocoModel
from src.model_stuff.downstream_model import MyDownstreamModel
from src.data_stuff.NEW_patch_dataset import PatchDataModule
from src.callback_stuff.PatientLevelValidation import PatientLevelValidation
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from rich import print
# from src.data_stuff.pip_tools import install
# install(["pytorch-lightning", "albumentations", "seaborn", "timm", "wandb", "plotly", "lightly"], quietly=True)


if __name__ == "__main__":
    print(f"🚙 Starting Downstream Experiment! 🚗")
    pl.seed_everything(42)

    # Data Dir and Model Checkpoint dir
    data_dir = "/workspace/repos/data/tcga_data_formatted/"
    model_save_path = "/workspace/repos/hrdl/saved_models/moco/{EXP_NAME}"
    embedder_checkpoint = "/workspace/repos/hrdl/saved_models/moco/.... wherever it is"

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--group_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()


    # make experiment name 
    EXP_NAME = f"downstream_gs{args.group_size}_bs{args.batch_size}_lr{args.learning_rate}_eps{args.num_epochs}"
    print(f"\tExperiment Name: {EXP_NAME}")

    # logger
    logger=WandbLogger(project="colorectal_cancer_ai", name=EXP_NAME)

    # callbacks
    patient_level_validation_callback = PatientLevelValidation(group_size=args.group_size, debug_mode=False)
    checkpoint_callback = ModelCheckpoint(
            dirpath=model_save_path,
            filename='{epoch}-{val_majority_vote_acc:.3f}-{val_acc_epoch:.3f}',
            save_top_k=1,
            verbose=True,
            monitor='val_majority_vote_acc',
            mode='max'
            )

    # Load moco checkpoint from stage 1
    embedder = MocoModel(args.moco_max_epochs).load_from_checkpoint(args.model_loc) 
    print("\tMoco Checkpoint Loaded ✅")

    # we only need its trained backbone
    backbone = embedder.feature_extractor.backbone

    # model
    model = MyDownstreamModel(
            backbone=backbone, 
            lr=args.learning_rate, 
            dataloader_group_size=args.group_size, 
            )
    print("\tDownstream Model Initialized ✅")

    # data
    dm = PatchDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            group_size=args.group_size,
            num_workers=os.cpu_count() 
            )
    print("\tDatamodule Initialized")

    trainer = Trainer(
            logger=logger,
            max_epochs=args.num_epochs,
            reload_dataloaders_every_n_epochs=1,
            callbacks=[checkpoint_callback, patient_level_validation_callback]
            )
    trainer.fit(model, dm)

# trainer.save_checkpoint("/workspace/repos/hrdl/saved_models/downstream/downstream10/{epoch}-{val_majority_vote_acc:.3f}-{val_acc_epoch:.3f}.pt")
