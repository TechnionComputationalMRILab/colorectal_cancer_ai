import argparse
import os
import tarfile
import hashlib

import torchvision

# https://github.com/fastai/imagenette

def check_args(resolution):
    possible_res = ["full_sz", "320px", "160px"]
    if (resolution not in possible_res): raise ValueError("--resolution arg must be in {possible_res}. You sent {resolution}.")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=str, default='full_sz')
    parser.add_argument('--download_dir', type=str, default='/home/shatz/repos/data/')
    args = parser.parse_args()

    check_args(args.resolution)

    #choose image sizes:
    datasets = {
        'full_sz': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz', # 1.5GB
        '320px': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz', # 326mb
        '160px': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz' # 94mb
    }
    
    data_dir = args.download_dir
    dataset_url = datasets[args.resolution]
    dataset_filename = dataset_url.split('/')[-1]
    dataset_foldername = dataset_filename.split('.')[0]
    dataset_filepath = os.path.join(data_dir,dataset_filename)
    dataset_folderpath = os.path.join(data_dir,dataset_foldername)
    os.makedirs(data_dir, exist_ok=True)
    download = False
    if not os.path.exists(dataset_filepath):
        download = True
    else:
        md5_hash = hashlib.md5()
        file = open(dataset_filepath, "rb")
        content = file.read()
        md5_hash.update(content)
        digest = md5_hash.hexdigest()
        if digest != 'fe2fc210e6bb7c5664d602c3cd71e612':
            download = True
    if download:
        print(f"⬇️  downloading imagenette...")
        torchvision.datasets.utils.download_url(dataset_url, data_dir)
        print("... done!")

    with tarfile.open(dataset_filepath, 'r:gz') as tar:
        tar.extractall(path=data_dir)

