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


class ImageFolderWithPaths(torchvision.datasets.DatasetFolder):
    """ 
    This class exists to train the downstream trainer on groups of patients at once.

    The idea is this:
    1. Choose how many patients (n_patients) and how many patches (n_patches) per patient you want in the batch.
    2. The batch size is effectively n_patients*n_patches
    3. Select "n_patients" random patients
    4. Find n_patches random patches per patient
    5. yield a list of patches for a patient with their label

    The dataloader will take care of how many patients I will use (n_patients == batch_size)
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolderWithPaths, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
        
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

