from src.data_stuff.pip_tools import install
install(["pytorch-lightning"], quietly=True)
from src.data_stuff.downstream_dataset import DownstreamTrainingDataset
import torchvision
import torch

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
train_transforms = torchvision.transforms.Compose([
    # torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(rgb_mean, rgb_std),
])
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(rgb_mean, rgb_std),
])

ds = DownstreamTrainingDataset(root_dir="/workspace/repos/TCGA/data/", transform=train_transforms)
item = ds[2]
print("sample shape:", item["data"].shape)

print(f"\t ... testing data loader ")
dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=4)
sample = next(iter(dl))
import pdb; pdb.set_trace()

# (Pdb) sample.keys()
# dict_keys(['label', 'patient_id', 'data_paths', 'data'])
# (Pdb) sample["label"]
# ['MSS', 'MSS']
# (Pdb) sample["patient_id"]
# ['TCGA-NH-A50U', 'TCGA-D5-5538']
# (Pdb) sample["data_paths"]
# [('/workspace/repos/TCGA/data/train/MSS/blk-TYIWFFWHNDRI-TCGA-NH-A50U-01Z-00-DX1.png', '/workspace/repos/TCGA/data/train/MSS/blk-WPMEQCDIYSVQ-TCGA-D5-5538-01Z-00-DX1.png'), ('/workspace/repos/TCGA/data/train/MSS/blk-VESTEPINDMST-TCGA-NH-A50U-01Z-00-DX1.png', '/workspace/repos/TCGA/data/train/MSS/blk-VRGENTTGCFCS-TCGA-D5-5538-01Z-00-DX1.png'), ('/workspace/repos/TCGA/data/train/MSS/blk-WRIDANMTISMQ-TCGA-NH-A50U-01Z-00-DX1.png', '/workspace/repos/TCGA/data/train/MSS/blk-YCGSSNTQSRLH-TCGA-D5-5538-01Z-00-DX1.png'), ('/workspace/repos/TCGA/data/train/MSS/blk-VEKKWATQPFKC-TCGA-NH-A50U-01Z-00-DX1.png', '/workspace/repos/TCGA/data/train/MSS/blk-VFWQESIMACND-TCGA-D5-5538-01Z-00-DX1.png'), ('/workspace/repos/TCGA/data/train/MSS/blk-WWKIRWPYTEGW-TCGA-NH-A50U-01Z-00-DX1.png', '/workspace/repos/TCGA/data/train/MSS/blk-TKTQEDRCLVIQ-TCGA-D5-5538-01Z-00-DX1.png')]
# (Pdb) len(sample["data_paths"])
# 5
# (Pdb) sample["data"].shape
# torch.Size([2, 5, 224, 224, 3])

print("--- done test ---")
