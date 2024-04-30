import os
import numpy as np
from datasets import Dataset as Dataset_HF, DatasetDict
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch



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
    
    

class SuperResolutionDataset_2(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.ToTensor()

        self.augmentation_transform_1 = transforms.Compose([
            # transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
        ])

        self.augmentation_transform_2 = transforms.Compose([
            # transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
        ])

        self.image_augmented = torch.empty((len(self.dataset)*3, 1, 64, 64))
        self.target_augmented = torch.empty((len(self.dataset)*3, 1, 128, 128))
        
        self.augment_dataset()

        del self.dataset
        

    def __len__(self):
        return len(self.image_augmented)
    

    def is_faulty_image(img: np.ndarray, threshold = 0.0005) -> bool:
        res1 = np.sum(np.where(img == 1, True, False))
        return res1/(img.shape[0]*img.shape[1]) > threshold


    def augment_dataset(self):

        for idx in range(len(self.dataset)):
            image = np.array(self.dataset[idx]["lr"], dtype=np.float32).transpose((1,2,0))
            target = np.array(self.dataset[idx]["hr"], dtype=np.float32).transpose((1,2,0))

            image = self.transform(image)
            target = self.transform(target)

            self.image_augmented[idx*3] = image
            self.target_augmented[idx*3] = target

            image_1 = self.augmentation_transform_1(image)
            target_1 = self.augmentation_transform_1(target)

            self.image_augmented[(idx*3)+1] = image_1
            self.target_augmented[(idx*3)+1] = target_1

            image_2 = self.augmentation_transform_2(image)
            target_2 = self.augmentation_transform_2(target)

            self.image_augmented[(idx*3)+2] = image_2
            self.target_augmented[(idx*3)+2] = target_2



    def __getitem__(self, idx):
        return self.image_augmented[idx], self.target_augmented[idx]