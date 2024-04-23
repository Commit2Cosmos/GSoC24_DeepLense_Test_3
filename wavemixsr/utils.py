import os
import numpy as np
from datasets import Dataset, DatasetDict


def train_test_eval_split(ds: Dataset):
    ds: DatasetDict = ds.train_test_split(
        test_size=0.2,
        seed=42
    )

    ds_train = ds['train']
    ds_test = ds['test']

    ds_test = ds_test.train_test_split(
        test_size=0.5,
        seed=42
    )

    ds_eval = ds_test['test']
    ds_test = ds_test['train']

    return ds_train, ds_test, ds_eval



def save_data(load_from_dir = "./datasets_lens", save_to_dir = "./datasets_lens"):

    lr_folder = os.path.join(load_from_dir, "LR")
    hr_folder = os.path.join(load_from_dir, "HR")


    lr_filenames = sorted(os.listdir(lr_folder))
    hr_filenames = sorted(os.listdir(hr_folder))


    data = {'lr': [], 'hr': []}
    for lr_filename, hr_filename in zip(lr_filenames, hr_filenames):
        if lr_filename.endswith('.npy') and hr_filename.endswith('.npy'):

            #* Load LR and HR images
            lr_image = np.load(os.path.join(lr_folder, lr_filename))
            hr_image = np.load(os.path.join(hr_folder, hr_filename))
            
            #* Append data to dictionary
            data['lr'].append(lr_image)
            data['hr'].append(hr_image)
        
        else:
            raise ValueError("Not .npy file found")


    ds = Dataset.from_dict(data)
    ds.save_to_disk(os.path.join(save_to_dir, "Lens"))