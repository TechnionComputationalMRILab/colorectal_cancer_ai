#!/usr/bin/env python
# coding: utf-8
from src.data_stuff.pip_tools import install
install(["torch==1.8.1","pytorch-lightning", "seaborn", "timm", "wandb", "plotly", "lightly"], quietly=True) #opencv-python too

import torch
print(torch.__version__)

