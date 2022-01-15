import numpy as np
import pandas as pd
import torch
import torchvision
from tqdm.notebook import tqdm
import glob
import re
from itertools import zip_longest
import random
# from skimage import io
from PIL import Image

class DownstreamTrainingDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None, dataset_type="train", group_size=5):
        """
        Args:
            root_dir (sting): Directory with all the data (eg '/tcmldrive/databases/Public/TCGA/data/')
            transform (callable, optional): Optional transform to be applied on a sample.
            dataset_type (string): "train" or "test"
        """
        print('\tInitializing Downstream Training Dataset...')
        self.transform = transform
        self.group_size = group_size
        self.classes = ["MSIMUT", "MSS"] #eventually make this inferred from folders
        self.dataset_type = dataset_type
        self.root_dir = root_dir
        self.train_dir = root_dir + 'train'
        self.val_dir = root_dir + 'test'
        self.all_filenames = self.get_all_file_paths()
        self.ultimate_re = r'train|test|TCGA-\w{2}-\w{4}|/MSIMUT/|/MSS/'
        self.dataset_dict_tcga = self.make_dataset_dict()
        self.check_dataset_dict(self.dataset_dict_tcga)
        self.index_mapping = self.__create_index_mapping__()
        print('\t... done initialization ✅')

    def __len__(self):
        return len(self.index_mapping)


    def __create_index_mapping__(self):
        """
        Maps the dataset_dict into a list that is indexable. Elements
        from this list will be yielded in __getitem__
        """
        train_index_mapping = []
        for class_label in self.classes:
            for patient_id in self.dataset_dict_tcga[self.dataset_type][class_label].keys():

                # split this list of paths in the patient into groups of n
                patient_patches_list = self.dataset_dict_tcga[self.dataset_type][class_label][patient_id]
                # shuffle list (equivalent to random.shuffle, except random.shuffle is in-place)
                patient_patches_list = random.sample(patient_patches_list, len(patient_patches_list))

                #https://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n
                grouped_list = list(zip_longest(*(iter(patient_patches_list),) * self.group_size))
                for group in grouped_list:
                    if None not in group:
                        train_index_mapping.append({
                            "label": class_label, 
                            "patient_id": patient_id, 
                            "data_paths": group 
                            })
        return train_index_mapping
            

    def __getitem__(self, idx):
        # idk why I need dis
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # looks like {'label': 'MSIMUT', 
        #               'patient_id': 'TCGA-CM-6171', 
        #               'data_paths': ('/workspace/repos/TCGA/data/train/MSIMUT/blk-THPNGVSFMQPH-TCGA-CM-6171-01Z-00-DX1.png', ... }
        patient_set = self.index_mapping[idx]

        # replace paths in patient set with torch tensor of all the patches
        # tensor should be (NxWxHxC)) where n is batch size (in this case it is self.group_size)
        patches = []
        for path in patient_set['data_paths']:
            patch = Image.open(path)
            patch = self.transform(patch).permute(1, 2, 0) #C,W,H->W,H,C
            patches.append(patch)
        patches_stack = torch.stack(patches)
        patient_set["data"] = patches_stack
        return patient_set


    def get_train_sample_filenames(self):
        """ filenames for all images in train dir"""
        train_img_filenames_MSIMUT = glob.glob(self.train_dir+'/MSIMUT/*.png')
        train_img_filenames_MSS = glob.glob(self.train_dir+'/MSS/*.png')
        all_train_filenames = train_img_filenames_MSIMUT + train_img_filenames_MSS
        return all_train_filenames

    def get_val_sample_filenames(self):
        """ filenames for all images in val dir"""
        test_img_filenames_MSIMUT = glob.glob(self.val_dir+'/MSIMUT/*.png')
        test_img_filenames_MSS = glob.glob(self.val_dir+'/MSS/*.png')
        all_val_filenames = test_img_filenames_MSIMUT + test_img_filenames_MSS
        return all_val_filenames

    def get_all_file_paths(self):
        """
        EX: ['/workspace/repos/TCGA/data/train/MSIMUT/blk-LISQHHKHDTVS-TCGA-CM-6171-01Z-00-DX1.png', ...]
        length is 192312
        """
        all_filenames = self.get_train_sample_filenames() + self.get_val_sample_filenames()
        return all_filenames

    def get_set_class_patientid(self, path):
        """
        This function will return the set (train/val), class(MSS/MSIMUT), and patientid for a path.
        EX: '/workspace/repos/TCGA/data/train/MSIMUT/blk-LISQHHKHDTVS-TCGA-CM-6171-01Z-00-DX1.png' -> ['train', '/MSIMUT/', 'TCGA-CM-6171']
        """
        matches = re.findall(self.ultimate_re, path)
        assert(len(matches)==3), f"There are {len(matches)} matches, but it should be 3"
        return matches



    def make_dataset_dict(self):
        """
        makes a nested dict following the structure below.
        """
        data_dict = {
                "train": {
                    "MSS": {
                        # patient_id: [],
                        # patient_id: [],
                        # ...
                        },
                    "MSIMUT": {}
                    },
                "test": {
                    "MSS": {},
                    "MSIMUT": {}
                    }
                }

        for i, path in enumerate(self.all_filenames):
            data_set, data_class, patient_id = self.get_set_class_patientid(path)
            data_class = data_class.replace('/', '') # remove "/" (/MSS/ -> MSS)
            if patient_id in data_dict[data_set][data_class].keys():
                data_dict[data_set][data_class][patient_id].append(path)
            else:
                data_dict[data_set][data_class][patient_id] = [path]
        return data_dict

    def check_dataset_dict(self, dataset_dict):
        mss_train_num_patients = len(dataset_dict["train"]["MSS"])
        msimut_train_num_patients = len(dataset_dict["train"]["MSIMUT"])
        mss_test_num_patients = len(dataset_dict["test"]["MSS"])
        msimut_test_num_patients = len(dataset_dict["test"]["MSIMUT"])
        f_str = (f"\n\t---\n"
                f"\tnum train mss patients      : {mss_train_num_patients}\n"
                f"\tnum test mss patients       : {mss_test_num_patients}\n"
                f"\tnum train msimut patients   : {msimut_train_num_patients}\n"
                f"\tnum test msimut patients   : {msimut_test_num_patients}\n"
                f"\t---\n")
        print(f_str)
