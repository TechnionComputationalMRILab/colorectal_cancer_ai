#!/usr/bin/env python
# coding: utf-8

import subprocess
import sys

def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("pytorch-lightning")
install("seaborn")
install("timm")

print("---- SWINT Exp (tcga) ----")

import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import re

from pytorch_lightning import LightningModule, Trainer
import torchvision.models as models
#from pl_bolts.callbacks import ORTCallback
import torchmetrics

# MY local imports
from src.data_stuff import dataset_tools
from src.callback_stuff import LogConfusionMatrix, PatientLevelValidation

from tqdm.notebook import tqdm


print('CUDA available:', torch.cuda.is_available())


# # ⬇️ Data

# ROOT_DIR = '/home/shatz/Documents/tcga_data/data/'
ROOT_DIR = '/workspace/repos/TCGA/data/'
TRAIN_DIR = ROOT_DIR + 'train'
TEST_DIR = ROOT_DIR + 'test'

class params:
    num_workers = 32
    bs = 128

### Transforms
rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
train_tfms = torchvision.transforms.Compose([
    # torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(rgb_mean, rgb_std),
])
test_tfms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(rgb_mean, rgb_std),
])

### Datasets
train_ds = dataset_tools.ImageFolderWithPaths(TRAIN_DIR, train_tfms)
test_ds = dataset_tools.ImageFolderWithPaths(TEST_DIR, test_tfms)

class_to_idx = train_ds.class_to_idx

# SUBSET FOR TESTING PURPOSES. DELETE LATER #
# train_ds = torch.utils.data.Subset(train_ds, np.random.randint(low=0, high=len(train_ds), size=int(len(train_ds)/8)))
# test_ds = torch.utils.data.Subset(test_ds, np.random.randint(low=0, high=len(test_ds), size=int(len(test_ds)/8)))

### Dataloaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=params.bs, num_workers=params.num_workers, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=params.bs, num_workers=params.num_workers, shuffle=False)


# In[ ]:


images = next(iter(train_dl))[1]
plt.imshow(torchvision.utils.make_grid(images, padding=20).permute(1, 2, 0))


# # Model

# In[ ]:


from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from src.model_stuff import MySwinTransformer


# In[ ]:


EXP_NAME = "tcga_SwinT"
logger = TensorBoardLogger("/workspace/repos/lightning_logs", name=EXP_NAME)


# In[ ]:


model = MySwinTransformer.MySwinTransformer(num_classes=2)
trainer = Trainer(gpus=1, max_epochs=50, logger=logger, callbacks=[
    LogConfusionMatrix.LogConfusionMatrix(class_to_idx=train_ds.class_to_idx),
    PatientLevelValidation.PatientLevelValidation()
])


# In[ ]:


trainer.fit(model, train_dataloader=train_dl, val_dataloaders=test_dl)


# In[ ]:


# !jupyter notebook stop 8888


# In[ ]:


# !touch done.txt

