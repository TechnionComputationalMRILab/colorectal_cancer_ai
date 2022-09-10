# 🚧 UNDER CONSTRUCTION 🚧


## Detecting Colorectal Cancer MSI/MSS Status From High Resolution WSI with Deep Learning
This repo contains code used in the paper "Patient-level Microsatellite Stability Assessment from Whole Slide Images By Combining Momentum Contrast Learning and Group Patch Embeddings" by Shats et al. 

## Dependencies
```
rich
wandb
numpy
pandas
PIL
pytorch
torchvision
torchmetrics
pytorch-lightning
lightly
```

## Relevant Files
`resnet_experiment.py` trains the baseline approach that was used for comparison against our method. It is our implemenation of the work in [Kather et al](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7423299/).

`moco_experiment.py` is used to train a patch feature extractor using [momentum contrast learning](https://arxiv.org/abs/2003.04297).

`downstream_experiment.py` uses the trained feature extractor from `moco_experiment.py` to now train a group-patch level MLP which can classify WSI's with higher accuracy.

