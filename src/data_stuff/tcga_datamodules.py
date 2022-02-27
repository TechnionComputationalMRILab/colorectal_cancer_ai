import pytorch_lightning as pl
import torch
import torchvision

# local imports
from ..data_stuff import dataset_tools

import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm.notebook import tqdm
import glob
import re


from torchvision.datasets.folder import default_loader
from torchvision.datasets.folder import Optional
from torchvision.datasets.folder import Callable
from torchvision.datasets.folder import Any
from torchvision.datasets.folder import Tuple
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS


class CustomTcgaDataset(torchvision.datasets.DatasetFolder):
    """	
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
        min_patches_per_patient (int): minimum number of patches a patient can have.
                                        if they have less, they are removed from samples idx.
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            min_patches_per_patient: int = 0,
    ):
        super(CustomTcgaDataset, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.min_patches_per_patient = min_patches_per_patient
        new_samples = []
        if min_patches_per_patient > 0:
            patient_filename_dict = self.get_patient_filename_dict(self.samples)
            remove_patients, remove_files = self.get_patients_and_files_with_less_than_n_patches(patient_filename_dict, self.min_patches_per_patient)
            if len(remove_patients) > 0:
                print("Patients that will be excluded from dataset:")
                print(remove_patients)
                for file, target in self.samples:
                    patient_regex = r'TCGA-\w{2}-\w{4}'
                    patient_id = re.findall(patient_regex, file)[0]
                    if patient_id in remove_patients:
                        print(f"\t- removing file {file}")
                    else:
                        new_samples.append((file, target))
                print(f'\t total num of samples removed: {len(self.samples)-len(new_samples)}')
                self.samples = new_samples
                self.imgs = self.samples
            else:
                print("\t --- no patients to remove")

            

    def get_patient_filename_dict(self, samples):
        """
        build a dict relating patients to filenames from the samples list that pytorch builds.
        the purpose (initially) is to remove all patients with less than "n" sample
        """
        patient_regex = r'TCGA-\w{2}-\w{4}'
        patient_filename_dict = {}
        for filename, target in samples:
            patient_id = re.findall(patient_regex, filename)[0]
            if patient_id in patient_filename_dict:
                patient_filename_dict[patient_id].append(filename)
            else:
                patient_filename_dict[patient_id] = [filename]
        return patient_filename_dict


    def get_patients_and_files_with_less_than_n_patches(self, patient_filename_dict, n):
        # n is min number of patches
        patients = []
        files = []
        for patient_id in patient_filename_dict.keys():
            if(len(patient_filename_dict[patient_id]) < n):
                patients.append(patient_id)
                files.extend(patient_filename_dict[patient_id])
        return patients, files
            

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (path, sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, sample, target


class TcgaDataModule(pl.LightningDataModule):
    def __init__(self,
            data_dir: str = "/workspace/repos/TCGA/data/",
            batch_size: int = 64,
            num_workers: int = 8,
            fast_subset: bool = True,
            min_patches_per_patient: int = 0):
            # custom_dataset: bool = False):

        super().__init__()
        self.data_dir = data_dir
        self.train_dir = self.data_dir + 'train'
        self.val_dir = self.data_dir + 'test'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fast_subset = fast_subset
        self.min_patches_per_patient = min_patches_per_patient
        # self.custom_dataset = custom_dataset

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
        if self.min_patches_per_patient>0:
            self.train_ds = CustomTcgaDataset(self.train_dir, self.train_transforms, min_patches_per_patient=8)
            self.val_ds   = CustomTcgaDataset(self.val_dir, self.val_transforms,min_patches_per_patient=8)
        else:
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

