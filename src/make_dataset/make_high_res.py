import argparse
import os
import numpy as np
# from tqdm import tqdm
from rich.progress import track
import torchvision
from sklearn.model_selection import train_test_split

import cv2

def make_highres(img_list, save_dir, out_res):
    """
    Takes a list of images and upscales them to out_res.
    Saves them in save_dir.
    """

    os.makedirs(save_dir, exist_ok=True)
    for i, filename in enumerate(track(img_list)):
        img = cv2.imread(filename)
        resized_img = cv2.resize(img, out_res)
        out_name = save_dir+str(i)+'.jpg'
        cv2.imwrite(out_name, resized_img)
    print(f"... finished writing {save_dir} ✅")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/shatz/repos/data/imagenette2/')
    parser.add_argument('--out_dir', type=str, default='/home/shatz/repos/data/imagenette2_hr/')
    parser.add_argument('--n_images', type=int, default=150)
    args = parser.parse_args()

    # folders I want to make super high res
    # all folders:
    # n01440764  FISH
    # n02102040  DOG
    # n02979186  
    # n03000684  
    # n03028079  
    # n03394916  
    # n03417042  
    # n03425413  
    # n03445777  
    # n03888257
    fish_folder = args.data_dir+"train/n01440764/"
    dog_folder = args.data_dir+"train/n02102040/"

    # how many imgs of each
    n_images = args.n_images
    out_res = (4000, 4000)
    val_sz = 0.3
    train_sz = 1-val_sz

    # get first n files from each dir (also need to append the folder name since its just files)
    fish_files = os.listdir(fish_folder)[:n_images]
    fish_files = [fish_folder+f for f in fish_files]
    dog_files = os.listdir(dog_folder)[:n_images]
    dog_files = [dog_folder+f for f in dog_files]

    # split images into train and test
    train_fish_files, val_fish_files = train_test_split(fish_files, train_size=train_sz, test_size=val_sz)
    train_dog_files, val_dog_files = train_test_split(dog_files, train_size=train_sz, test_size=val_sz)

    # make them high res as save them next to the data_dir
    train_highres_fish_dir = args.out_dir+"train/fish/"
    train_highres_dog_dir = args.out_dir+"train/dog/"
    val_highres_fish_dir = args.out_dir+"val/fish/"
    val_highres_dog_dir = args.out_dir+"val/dog/"
    make_highres(train_fish_files, save_dir=train_highres_fish_dir,  out_res=out_res)
    make_highres(train_dog_files, save_dir=train_highres_dog_dir, out_res=out_res)
    make_highres(val_fish_files, save_dir=val_highres_fish_dir, out_res=out_res)
    make_highres(val_dog_files, save_dir=val_highres_dog_dir, out_res=out_res)

