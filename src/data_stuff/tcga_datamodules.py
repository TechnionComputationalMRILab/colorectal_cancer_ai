import pytorch_lightning as pl
import torch
import torchvision

# local imports
from ..data_stuff import dataset_tools

class TcgaDataModule(pl.LightningDataModule):
    def __init__(self,
            data_dir: str = "/workspace/repos/TCGA/data/",
            batch_size: int = 64,
            num_workers: int = 16,
            fast_subset: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.train_dir = self.data_dir + 'train'
        self.val_dir = self.data_dir + 'test'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_to_idx = self.get_class_to_idx_dict()


    def prepare_data(self):
        # things to do on 1 gpu

        rgb_mean = (0.4914, 0.4822, 0.4465)
        rgb_std = (0.2023, 0.1994, 0.2010)
        self.train_transforms = torchvision.transforms.Compose([
            # torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(rgb_mean, rgb_std),
        ])
        self.val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(rgb_mean, rgb_std),
        ])
        

    def setup(self, stage):
        # things to do on every accelerator (distibuted mode)
        # splits, etc
        self.train_ds = dataset_tools.ImageFolderWithPaths(self.train_dir, self.train_transforms)
        self.val_ds   = dataset_tools.ImageFolderWithPaths(self.val_dir, self.val_transforms)


    def train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                )
        return train_dataloader


    def val_dataloader(self):
        val_dataloader = torch.utils.data.DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                )
        return val_dataloader

    def get_class_to_idx_dict(self):
        """
        make dummy dataset just to get the class to idx dict.
        - this is currently needed for LogConfusionMatrix callback
        """
        class_to_idx = dataset_tools.ImageFolderWithPaths(self.train_dir).class_to_idx
        return class_to_idx
