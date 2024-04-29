import os
import numpy as np
from datasets import Dataset as Dataset_HF, DatasetDict
from torch.utils.data import Dataset
import torchvision.transforms as transforms



def train_test_eval_split(ds: Dataset_HF):
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

    lr_folder = os.path.join(load_from_dir, "LR_2")
    hr_folder = os.path.join(load_from_dir, "HR_2")


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


    ds = Dataset_HF.from_dict(data)
    ds.save_to_disk(os.path.join(save_to_dir, "Lens_2"))


class SuperResolutionDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.transform = transforms.ToTensor()
            

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image = self.dataset[idx]["lr"]
            image = np.array(image, dtype=np.float32).transpose((1,2,0))


            target = self.dataset[idx]["hr"] 
            target = np.array(target, dtype=np.float32).transpose((1,2,0))

            image = self.transform(image)
            target = self.transform(target)

            return image, target