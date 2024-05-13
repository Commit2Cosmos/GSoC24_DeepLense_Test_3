import os
import numpy as np
from datasets import Dataset as Dataset_HF, DatasetDict
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from scipy.signal import correlate
import time
import torch.nn.functional as F



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
    for (i, (lr_filename, hr_filename)) in enumerate(zip(lr_filenames, hr_filenames)):
        if i == 23 or i == 103:
            continue
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
    

class Wiener:
    def __call__(self, im: torch.Tensor, mysize=None, noise=None) -> torch.Tensor:
        
        im = np.asarray(im)
        if mysize is None:
            mysize = [4] * im.ndim
        mysize = np.asarray(mysize)

        if mysize.shape == ():
            mysize = np.repeat(mysize.item(), im.ndim)

        # Estimate the local mean
        lMean = correlate(im, np.ones(mysize), 'same') / np.prod(mysize, axis=0)

        # Estimate the local variance
        lVar = (correlate(im ** 2, np.ones(mysize), 'same') /
            np.prod(mysize, axis=0) - lMean ** 2)

        # Estimate the noise power if needed.
        if noise is None:
            noise = np.mean(np.ravel(lVar), axis=0)

        res = (im - lMean)
        res *= (1 - noise / lVar)
        res += lMean
        out = np.where(lVar < noise, lMean, res)

        return torch.tensor(out, dtype=torch.float32)


def cutblur(im1, im2, prob, alpha):
    if im1.size() != im2.size():
        raise ValueError("im1 and im2 have to be the same resolution.")

    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = int(h*cut_ratio), int(w*cut_ratio)
    cy = np.random.randint(0, h-ch+1)
    cx = np.random.randint(0, w-cw+1)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        im2[..., cy:cy+ch, cx:cx+cw] = im1[..., cy:cy+ch, cx:cx+cw]
    else:
        im2_aug = im1.clone()
        im2_aug[..., cy:cy+ch, cx:cx+cw] = im2[..., cy:cy+ch, cx:cx+cw]
        im2 = im2_aug

    return im1, im2


def apply_cutblur(HR, LR, prob=1.0, alpha=0.4):

    if HR.size() != LR.size():
        LR = F.interpolate(LR, scale_factor=2, mode="nearest")

    HR, LR = cutblur(
        HR.clone(), LR.clone(),
        prob=prob, alpha=alpha
    )

    return HR, LR



class MinMaxNormalizeImage:
    def __call__(self, img: torch.Tensor):
        min_val = img.min()
        max_val = img.max()
        normalized_tensor = (img - min_val) / (max_val - min_val)
        return normalized_tensor


class SuperResolutionDataset_2(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            Wiener(),
            MinMaxNormalizeImage()
        ])

        self.augmentation_transform_1 = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(45, scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
            Wiener(),
            MinMaxNormalizeImage()
        ])

        self.augmentation_transform_2 = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(45, scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
            Wiener(),
            MinMaxNormalizeImage()
        ])

        self.image_augmented = torch.empty((len(self.dataset)*3, 1, 64, 64))
        self.target_augmented = torch.empty((len(self.dataset)*3, 1, 128, 128))
        
        self.augment_dataset()

        del self.dataset
        

    def __len__(self):
        return len(self.image_augmented)
    

    def augment_dataset(self):

        for idx in range(len(self.dataset)):
            image = np.array(self.dataset[idx]["lr"], dtype=np.float32).transpose((1,2,0))
            target = np.array(self.dataset[idx]["hr"], dtype=np.float32).transpose((1,2,0))

            image = self.transform(image)
            target = self.transform(target)

            self.image_augmented[idx*3] = image
            self.target_augmented[idx*3] = target

            torch.manual_seed(idx)
            image_1 = self.augmentation_transform_1(image)
            torch.manual_seed(idx)
            target_1 = self.augmentation_transform_1(target)

            self.image_augmented[(idx*3)+1] = image_1
            self.target_augmented[(idx*3)+1] = target_1

            torch.manual_seed(idx+42)
            image_2 = self.augmentation_transform_2(image)
            torch.manual_seed(idx+42)
            target_2 = self.augmentation_transform_2(target)

            self.image_augmented[(idx*3)+2] = image_2
            self.target_augmented[(idx*3)+2] = target_2



    def __getitem__(self, idx):
        return self.image_augmented[idx], self.target_augmented[idx]