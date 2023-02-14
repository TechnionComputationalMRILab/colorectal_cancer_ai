import argparse
import os
import re
from rich.progress import track
import numpy as np
import cv2

def write_files(patient_dict, out_dir):
    for patient_name in track(patient_dict.keys()):
        os.makedirs(os.path.join(out_dir, patient_name))
        for file_name in patient_dict[patient_name]:
            file_basename = os.path.basename(file_name)
            img = cv2.imread(file_name)
            new_path = os.path.join(out_dir, patient_name, file_basename)
            cv2.imwrite(new_path, img)

def check_patient_dict(patient_dict):
    num_patients = len(patient_dict)
    num_patches_list = sorted([len(vs) for vs in patient_dict.values()])
    total_num_patches = sum(num_patches_list)
    avg_num_patches = np.mean(num_patches_list)
    least_patches = num_patches_list[:10]
    most_patches = num_patches_list[-10:]
    print("-- 🧾 checking 🧾 --")
    print(f"\tFound {num_patients} patients.")
    print(f"\tTotal num patches: {total_num_patches}")
    print(f"\tAvg   num patches: {avg_num_patches}")
    print(f"\tLeast num patches: {least_patches}")
    print(f"\tMost num patches: {most_patches}")


def build_patient_dict(files_list):
    patient_re = r'TCGA-\w{2}-\w{4}'
    patient_dict = {}
    for f in track(files_list):
        patient_name = re.findall(patient_re, f)[0]
        if patient_name in patient_dict:
            patient_dict[patient_name].append(f)
        else:
            patient_dict[patient_name] = [f]
    check_patient_dict(patient_dict)
    return patient_dict
        

def format_dir(data_dir, out_dir):
    print(f'🛠 Formatting {data_dir} into {out_dir} 🔩')

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]

    patient_dict = build_patient_dict(files)
    write_files(patient_dict, out_dir)
    print('✅ Done')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/shatz/repos/data/tcga_data/data/')
    parser.add_argument('--out_dir', type=str, default='/home/shatz/repos/data/tcga_data_formatted/')
    args = parser.parse_args()

    # input dirs
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    train_mss_dir = os.path.join(train_dir, "MSS")
    train_msimut_dir = os.path.join(train_dir, "MSIMUT")
    test_mss_dir = os.path.join(test_dir, "MSS")
    test_msimut_dir = os.path.join(test_dir, "MSIMUT")

    # output dirs
    out_train_dir = os.path.join(args.out_dir, "train")
    out_test_dir = os.path.join(args.out_dir, "test")
    out_train_mss_dir = os.path.join(out_train_dir, "MSS")
    out_train_msimut_dir = os.path.join(out_train_dir, "MSIMUT")
    out_test_mss_dir = os.path.join(out_test_dir, "MSS")
    out_test_msimut_dir = os.path.join(out_test_dir, "MSIMUT")

    # Format
    format_dir(train_mss_dir, out_train_mss_dir)
    format_dir(train_msimut_dir, out_train_msimut_dir)
    format_dir(test_mss_dir, out_test_mss_dir)
    format_dir(test_msimut_dir, out_test_msimut_dir)
