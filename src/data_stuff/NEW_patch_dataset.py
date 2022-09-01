import os
import torch
import torchvision
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset
from itertools import zip_longest
from PIL import Image
import random
import copy
"""

This file contains both a torch dataset as well as a pytorch lightning datamodule (below)

"""
class PatchDataset(Dataset):
    """
    Generic dataloader where images are arranged in this way:
    --------------------------------------------------------
    root/dog/dog1/patch1.jpg
    root/dog/dog1/patch2.jpg
    ...
    root/dog/dog873/patch44.jpg

    root/fish/fish1/patch1.jpg
    root/fish/fish1/patch2.jpg
    ...
    root/fish/fish999/patch823.jpg
    
    """

    def __init__(self, root_dir, transform=None, group_size=1):
        self.root_dir = root_dir
        self.transform = transform
        self.group_size = group_size 

        self.samples, self.class_to_idx = self.make_dataset()
        print(f"🏛  -- class to index: {self.class_to_idx}")
        self.remove_images_with_few_patches(self.group_size, self.samples)
        self.grouped_samples = self.group_dataset(self.samples, self.group_size)

        # to check for dataset reloading
        self.random_id_state = random.randrange(30)
        print(self.random_id_state)

    def remove_images_with_few_patches(self, min_patches, dataset_dict):
        for k in dataset_dict.copy().keys():
            num_patches = len(dataset_dict[k][0])
            if num_patches < min_patches:
                dataset_dict.pop(k, None)
                print(f"Removed {k}, {num_patches} patches")
        

    def make_dataset(self):
        """
        Returns a Dict representing the file structure of the train or val folder
        Also returns the class to index dict

        Ex:
        {
            fish1: ([root/fish/fish1/patch1.jpg, root/fish/fish1/patch2.jpg, ... root/...patch323.jpg], 1),
            fish2: ([root/fish/fish2/patch2.jpg, root/fish/fish2/patch2.jpg, ... root/...patch9899.jpg], 1),
            ...
            dog1: ([root/dog/dog1/patch1.jpg, root/dog/dog1/patch2.jpg, ... root/...patch999.jpg], 0),
            ...
        }

        General structure of above dict:
        { image_id: ([list of corresponding patches], label), }
        """

        # make class to index dict
        classes_list = sorted(os.listdir(self.root_dir))
        class_to_idx = {}
        for idx, class_name in enumerate(classes_list):
            class_to_idx[class_name] = torch.tensor(idx)
        
        # make dataset
        dataset_dict = {}
        for class_name in classes_list:
            for patch_folder in os.listdir(os.path.join(self.root_dir, class_name)):
                patch_paths = []
                for patch in os.listdir(os.path.join(self.root_dir, class_name, patch_folder)):
                    full_patch_path = os.path.join(self.root_dir, class_name, patch_folder, patch)
                    patch_paths.append(full_patch_path)
                dataset_dict[patch_folder] = (patch_paths, class_to_idx[class_name])

        return dataset_dict, class_to_idx
    
    def group_dataset(self, dataset_dict, group_size):
        """
        Take the dataset_dict made in make_dataset and split based on group_size.

        Inputs:
            dataset_dict: formatted dataset from self.make_dataset()
            group_size: number of patches per image_id

        Returns:
            grouped_dataset: list where each index is a sample that will be returned from __getitem__
                            each index follows: (image_id, [list_of_images_with_len_group_size], label )
        """
        grouped_dataset = []
        grouped_dataset = []
        for image_id in dataset_dict.keys():
            patches_list, label = dataset_dict[image_id]
            #https://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n
            random.shuffle(patches_list) # so that the groups are different every epoch (NEED TO  RE-init dataset)
            grouped_list = list(zip_longest(*(iter(patches_list),) * group_size))
            for image_list in grouped_list:
                if None not in image_list:
                    individual_sample = (image_id, ",".join(image_list), label)
                    grouped_dataset.append(individual_sample)
                else: # None is there
                    # need to get some random random patches to fill the nones
                    not_none_image_list = []
                    for img in image_list:
                        if img is not None:
                            not_none_image_list.append(img)
                        else:
                            random_idx = random.randrange(len(image_list))
                            random_img = patches_list[random_idx]
                            not_none_image_list.append(random_img)
                    not_none_image_list = tuple(not_none_image_list)
                    individual_sample = (image_id, ",".join(not_none_image_list), label)
                    grouped_dataset.append(individual_sample)
        return grouped_dataset
            


    def __len__(self):
        return len(self.grouped_samples)

    def __getitem__(self, index):
        sample = self.grouped_samples[index]
        # sample = ('img99852', 'paths/to/all,associated/images,/separated/by/commas', tensor(0))

        # add images
        img_id, image_paths, label = sample
        image_paths = image_paths.split(",")
        patches = []
        for image_path in image_paths:
            patch = Image.open(image_path)
            patch = self.transform(patch)
            # patch = self.transform(patch).permute(1, 2, 0) #C,W,H->W,H,C
            patches.append(patch)
        patches_stack = torch.stack(patches)

        if self.group_size > 1:
            image_paths, patches_stack = self.shuffle_tings(image_paths, patches_stack)
            image_paths = ",".join(image_paths)
        return img_id, image_paths, label, patches_stack

    def shuffle_tings(self, paths_list, patches_stack):
        idxs = torch.randperm(len(paths_list))
        shuffled_paths_list = list(np.array(paths_list)[idxs])
        shuffled_patches_stack = patches_stack[idxs]
        return shuffled_paths_list, shuffled_patches_stack,  

    def get_samples_dict(self):
        # return the self.samples dict which looks like the depiction in self.make_dataset()
        return self.samples

    def get_label_dict(self):
        # returns a dict that correspons image/folder id to label e.g.:
        # {img_id: label}
        label_dict = {}
        for img_id in self.samples.keys():
            label_dict[img_id] = self.samples[img_id][1]
        return label_dict

    def get_img_samples_score_dict(self):
        # returns a dict of dicts where first you index by img_id, then patch_id, and you must put in the score manually
        img_samples_score_dict = {}
        for img_id in self.samples.keys():
            img_samples_score_dict[img_id] = {}
            for sample_path in self.samples[img_id][0]: # 0 bc self.samples["img432"] = tup([list/of, pathss/], label(0))
                img_samples_score_dict[img_id][sample_path] = None
        return copy.deepcopy(img_samples_score_dict)


########################## Done Dataset. Datamodule below ##################################





class PatchDataModule(pl.LightningDataModule):
    def __init__(self,
            data_dir: str = "/home/shatz/repos/data/imagenette_tesselated/",
            batch_size: int = 64,
            group_size: int = 1,
            num_workers: int = 4,
            ):
        super().__init__()
        self.data_dir = data_dir
        self.train_dir = self.data_dir + 'train'
        self.val_dir = self.data_dir + 'val'
        self.batch_size = batch_size
        self.group_size = group_size
        self.num_workers = num_workers


    def prepare_data(self):
        # things to do on 1 gpu

        rgb_mean = (0.4914, 0.4822, 0.4465)
        rgb_std = (0.2023, 0.1994, 0.2010)
        self.train_tfms = torchvision.transforms.Compose([
            # torchvision.transforms.RandomCrop(32, padding=4),
            # torchvision.transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(rgb_mean, rgb_std),
        ])
        self.val_tfms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(rgb_mean, rgb_std),
        ])

    def setup(self, stage=None):
        # things to do on every accelerator (distibuted mode)
        # splits, etc
        self.train_ds = PatchDataset(self.train_dir, group_size=self.group_size, transform=self.train_tfms)
        self.val_ds = PatchDataset(self.val_dir, group_size=self.group_size, transform=self.val_tfms)


    def train_dataloader(self):
        self.train_ds = PatchDataset(self.train_dir, group_size=self.group_size, transform=self.train_tfms)
        train_dataloader = torch.utils.data.DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                # collate_fn=self.collate_fn,
                drop_last=True
                )
        return train_dataloader


    def val_dataloader(self):
        # self.val_ds = PatchDataset(self.val_dir, group_size=self.group_size, transform=self.val_tfms)
        val_dataloader = torch.utils.data.DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                drop_last=True
                )
        return val_dataloader
