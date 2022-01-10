import pytorch_lightning as pl
import torch
import torchvision
import os
import tarfile
import hashlib

# https://github.com/fastai/imagenette

# local imports
from ..data_stuff import dataset_tools

class ImagenetteDataModule(pl.LightningDataModule):
    def __init__(self,
            data_dir: str = "/workspace/repos/imagenette_data",
            batch_size: int = 64,
            num_workers: int = 8,
            fast_subset: bool = False):
        super().__init__()

        #choose image sizes:
        datasets = {
            'full_sz': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz', # 1.5GB
            '320px': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz', # 326mb
            '160px': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz' # 94mb
        }

        dataset_url = datasets['320px']
        dataset_filename = dataset_url.split('/')[-1]
        dataset_foldername = dataset_filename.split('.')[0]
        dataset_filepath = os.path.join(data_dir,dataset_filename)
        dataset_folderpath = os.path.join(data_dir,dataset_foldername)
        os.makedirs(data_dir, exist_ok=True)
        download = False
        if not os.path.exists(dataset_filepath):
            download = True
        else:
            md5_hash = hashlib.md5()
            file = open(dataset_filepath, "rb")
            content = file.read()
            md5_hash.update(content)
            digest = md5_hash.hexdigest()
            if digest != 'fe2fc210e6bb7c5664d602c3cd71e612':
                download = True
        if download:
            print(f"⬇️  downloading imagenette...")
            torchvision.datasets.utils.download_url(dataset_url, data_dir)
            print("... done!")

        with tarfile.open(dataset_filepath, 'r:gz') as tar:
            tar.extractall(path=data_dir)

        self.data_dir = dataset_folderpath
        self.train_dir = self.data_dir + '/train'
        self.val_dir = self.data_dir + '/val'
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # things to do on 1 gpu

        rgb_mean = (0.4914, 0.4822, 0.4465)
        rgb_std = (0.2023, 0.1994, 0.2010)
        self.train_transforms = torchvision.transforms.Compose([
            # torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(rgb_mean, rgb_std),
        ])
        self.val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
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

