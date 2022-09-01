import argparse
import os
import numpy as np
# from tqdm import tqdm
from rich.progress import track
import torchvision
from sklearn.model_selection import train_test_split

import cv2

def random_crop(img, crop_sz):
    """
    Random crop of any numpy image
    """
    max_x = img.shape[1] - crop_sz[1]
    max_y = img.shape[0] - crop_sz[0]
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop = img[y: y + crop_sz[0], x: x + crop_sz[1]]
    return crop


def tesselate_image(img_path, n_crops, crop_sz):
    img = cv2.imread(img_path)
    crops = []
    for n in range(n_crops):
        crop = random_crop(img, crop_sz)
        crops.append(crop)
    return crops


def tesselate_folder(data_dir, destination_dir, n_crops, crop_sz, num_list):
    img_names = os.listdir(data_dir)
    for i, img_name in enumerate(track(img_names)):
        unique_id = num_list.pop()
        dest_img_dir = destination_dir + f'img{unique_id}/'
        os.makedirs(dest_img_dir, exist_ok=True)
        full_img_path = data_dir + img_name
        crops = tesselate_image(full_img_path, n_crops, crop_sz)
        for c, crop in enumerate(crops):
            img_destination = dest_img_dir + f'crop{c}' + '.jpg'
            cv2.imwrite(img_destination, crop)
    print(f"... done tesselating {data_dir} into {destination_dir} ✅")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/shatz/repos/data/imagenette2_hr/')
    parser.add_argument('--out_dir', type=str, default='/home/shatz/repos/data/imagenette2_tesselated/')
    args = parser.parse_args()

    # folders I want to tesselate
    train_fish_folder = args.data_dir+ 'train/fish/'
    train_dog_folder = args.data_dir+ 'train/dog/'
    val_fish_folder = args.data_dir+ 'val/fish/'
    val_dog_folder = args.data_dir+ 'val/dog/'

    # where I want the tesselations to go (dest folders)
    dest_train_fish_folder = args.out_dir + 'train/fish/'
    dest_train_dog_folder = args.out_dir + 'train/dog/'
    dest_val_fish_folder = args.out_dir + 'val/fish/'
    dest_val_dog_folder = args.out_dir + 'val/dog/'

    # tesselation/cropping hypers
    n_tesselations = 40
    tesselation_sz = (224, 224)

    # make destination dirs
    os.makedirs(dest_train_fish_folder, exist_ok=True)
    os.makedirs(dest_train_dog_folder, exist_ok=True)
    os.makedirs(dest_val_fish_folder, exist_ok=True)
    os.makedirs(dest_val_dog_folder, exist_ok=True)

    # list to ensure image names are unique
    num_list = list(range(100000))

    # make crops
    tesselate_folder(train_fish_folder, dest_train_fish_folder, n_tesselations, tesselation_sz, num_list)
    tesselate_folder(train_dog_folder, dest_train_dog_folder, n_tesselations, tesselation_sz, num_list)
    tesselate_folder(val_fish_folder, dest_val_fish_folder, n_tesselations, tesselation_sz, num_list)
    tesselate_folder(val_dog_folder, dest_val_dog_folder, n_tesselations, tesselation_sz, num_list)

    

