import pytorch_lightning as pl
import torch
import torchvision

# local imports
from ..data_stuff import dataset_tools

class TcgaDataModule(pl.LightningDataModule):
    def __init__(self,
            data_dir: str = "/workspace/repos/TCGA/data/",
            batch_size: int = 64,
            num_workers: int = 8,
            fast_subset: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.train_dir = self.data_dir + 'train'
        self.val_dir = self.data_dir + 'test'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fast_subset = fast_subset

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
        if self.fast_subset:
            train_idxs = list(range(int(len(self.train_ds)/2)))
            val_idxs = list(range(int(len(self.val_ds)/2)))
            self.train_ds = torch.utils.data.Subset(self.train_ds, train_idxs)
            self.val_ds = torch.utils.data.Subset(self.val_ds, val_idxs)

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
        Returns something like {'MSIMUT': 0, 'MSS': 1}
        I have to make a dummy dataset because setup() doesnt run in init(), so the self.train_ds
        is not available at time of initialization :(. There are better ways of doing it but I dont
        really care.
        """
        class_to_idx = dataset_tools.ImageFolderWithPaths(self.train_dir).class_to_idx
        return class_to_idx

    def get_idx_to_class_dict(self):
        """ 
        Returns something like {0: 'MSIMUT', 1: 'MSS'} 
        """
        class_to_idx_dict = self.get_class_to_idx_dict() # {'MSIMUT': 0, 'MSS': 1}
        idx_to_class_dict = {v: k for k, v in class_to_idx_dict.items()} # {0: 'MSIMUT', 1: 'MSS'}
        return idx_to_class_dict

