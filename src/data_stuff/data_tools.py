import pandas as pd
import glob
import re

ROOT_DIR = '/tcmldrive/databases/Public/TCGA/data/'
TRAIN_DIR = ROOT_DIR + 'train'
TEST_DIR = ROOT_DIR + 'test'
clinical_atlas_path = ROOT_DIR + 'clinical_Atlas.csv'
data_table_colorectal_adenocarcinoma_path = ROOT_DIR + 'data_table_Colorectal_Adenocarcinoma.csv'

def get_patient_name_from_path(path:str):
    path_pattern = r'TCGA-\w{2}-\w{4}'
    # Matches look like: ['TCGA-AZ-5403']
    patient_name = re.findall(path_pattern, path)
    assert(len(patient_name) == 1), "🛑 more than 1 patient name found"
    return patient_name[0]
    

def get_dfs():
    """ returns atlas_df and adeno_df """

    atlas_df = pd.read_csv(clinical_atlas_path)
    aden_df = pd.read_csv(data_table_colorectal_adenocarcinoma_path)
    return atlas_df, aden_df


def get_patients():
    """
    returns all the patients from atlas and aden files.
    Not all of the patients were used in our dataset.
    Actually most were not.
    """

    atlas_df, aden_df = get_dfs()

    atlas_patients = list(atlas_df["TCGA Participant Barcode"])
    aden_patients = list(aden_df["patient"])

    all_patients = atlas_patients + aden_patients

    return all_patients


def get_train_sample_filenames():
    """ filenames for all images in train dir"""

    train_img_filenames_MSIMUT = glob.glob(TRAIN_DIR+'/MSIMUT/*.png')
    train_img_filenames_MSS = glob.glob(TRAIN_DIR+'/MSS/*.png')
    all_train_filenames = train_img_filenames_MSIMUT + train_img_filenames_MSS
    return all_train_filenames


def get_test_sample_filenames():
    """ filenames for all images in val dir"""

    test_img_filenames_MSIMUT = glob.glob(TEST_DIR+'/MSIMUT/*.png')
    test_img_filenames_MSS = glob.glob(TEST_DIR+'/MSS/*.png')
    all_test_filenames = test_img_filenames_MSIMUT + test_img_filenames_MSS
    return all_test_filenames

def get_all_sample_filenames():
    """ Get all samples from train and test """

    train_filenames = get_train_sample_filenames()
    test_filenames = get_test_sample_filenames()
    return train_filenames + test_filenames

def get_all_patients_from_data_filenames():
    """ return a list of all the patient names present in the data filenames """
    
    all_sample_filenames = get_all_sample_filenames()
    pattern = r'TCGA-\w{2}-\w{4}'
    patients = [re.findall(pattern, filename)[0] for filename in all_sample_filenames]
    return patients


def get_train_patients_and_class_from_filenames():
    """ returns dict {patient_id: class} for patients found in train dict """

    patients_class_dict = {}
    train_sample_filenames = get_train_sample_filenames()
    pattern = r'TCGA-\w{2}-\w{4}|/MSIMUT/|/MSS/'
    for filename in train_sample_filenames:
        match = re.findall(pattern, filename)
        if len(match) > 2:
            print("ERROR contains more than 2 regex matches:", filename)
        else:
            target = match[0][1:-1] # clip off then ends since they are "/" from the regex
            patient_id = match[1]
            if patient_id in patients_class_dict:
                # just make sure the class hasnt changed
                assert(patients_class_dict[patient_id] == target)
            else:
                patients_class_dict[patient_id] = target
    return patients_class_dict   


def get_test_patients_and_class_from_filenames():
    """ returns dict {patient_id: class} for patients found in test dict """

    patients_class_dict = {}
    test_sample_filenames = get_test_sample_filenames()
    pattern = r'TCGA-\w{2}-\w{4}|/MSIMUT/|/MSS/'
    for filename in test_sample_filenames:
        match = re.findall(pattern, filename)
        if len(match) > 2:
            print("ERROR contains more than 2 regex matches:", filename)
        else:
            target = match[0][1:-1] # clip off then ends since they are "/" from the regex
            patient_id = match[1]
            if patient_id in patients_class_dict:
                # just make sure the class hasnt changed
                assert(patients_class_dict[patient_id] == target)
            else:
                patients_class_dict[patient_id] = target
    return patients_class_dict   


def get_all_patients_and_class_from_filenames():
    """ returns a dict {patient_id: class} for all patient names present in the image filenames """

    patients_class_dict = {}
    all_sample_filenames = get_all_sample_filenames()
    pattern = r'TCGA-\w{2}-\w{4}|/MSIMUT/|/MSS/'
    for filename in all_sample_filenames:
        match = re.findall(pattern, filename)
        if len(match) > 2:
            print("ERROR contains more than 2 regex matches:", filename)
        else:
            target = match[0][1:-1] # clip off then ends since they are "/" from the regex
            patient_id = match[1]
            if patient_id in patients_class_dict:
                # just make sure the class hasnt changed
                assert(patients_class_dict[patient_id] == target)
            else:
                patients_class_dict[patient_id] = target
    return patients_class_dict


def get_train_patients_and_files():
    """ returns a dict of {patient_id: [img1, img2, ... imgn]} """
    
    patients_files_dict = {}
    train_sample_filenames = get_train_sample_filenames()
    train_patients = get_train_patients_and_class_from_filenames().keys()
    for patient in train_patients:
        patient_files = []
        for f_name in train_sample_filenames:
            if patient in f_name:
                patient_files.append(f_name)
        patients_files_dict[patient] = patient_files
    return patients_files_dict


def get_test_patients_and_files():
    """ returns a dict of {patient_id: [img1, img2, ... imgn]} """
    
    patients_files_dict = {}
    test_sample_filenames = get_test_sample_filenames()
    test_patients = get_test_patients_and_class_from_filenames().keys()
    for patient in test_patients:
        patient_files = []
        for f_name in test_sample_filenames:
            if patient in f_name:
                patient_files.append(f_name)
        patients_files_dict[patient] = patient_files
    return patients_files_dict


def train_data_dataframe():
    """ returns dataframe with each row relating patient, class, and all image filenames """
    
    patient_filenames_dict = get_train_patients_and_files()
    patient_class_dict = get_train_patients_and_class_from_filenames()

    df = pd.DataFrame({"patient_id":pd.Series(dtype='str'),
                     "class":pd.Series(dtype="str"),
                     "filenames":pd.Series(dtype='object')})
    df["patient_id"] = patient_filenames_dict.keys()
    df["class"] = df.apply(lambda x: patient_class_dict[x["patient_id"]], axis=1)
    df["filenames"] = df.apply(lambda x: patient_filenames_dict[x["patient_id"]], axis=1)
    
    return df

def test_data_dataframe():
    """ returns dataframe with each row relating patient, class, and all image filenames """
    
    patient_filenames_dict = get_test_patients_and_files()
    patient_class_dict = get_test_patients_and_class_from_filenames()

    df = pd.DataFrame({"patient_id":pd.Series(dtype='str'),
                     "class":pd.Series(dtype="str"),
                     "filenames":pd.Series(dtype='object')})
    df["patient_id"] = patient_filenames_dict.keys()
    df["class"] = df.apply(lambda x: patient_class_dict[x["patient_id"]], axis=1)
    df["filenames"] = df.apply(lambda x: patient_filenames_dict[x["patient_id"]], axis=1)
    
    return df
