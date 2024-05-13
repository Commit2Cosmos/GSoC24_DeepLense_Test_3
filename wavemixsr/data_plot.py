from datasets import Dataset as Dataset_HF
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from utils import SuperResolutionDataset, SuperResolutionDataset_2, apply_cutblur
import torch


def plot_imgs(dataset: SuperResolutionDataset, j, cols):
    fig, axes = plt.subplots(2, cols, sharex='all', sharey='all', figsize=(14,9))
    plt.axis('off')

    axes = axes.flatten()
    
    for i in range(0, int(len(axes)/2)):

        HR = dataset[i+j][1].unsqueeze(0)
        LR = dataset[i+j][0].unsqueeze(0)

        HR, LR = apply_cutblur(HR, LR)

        axes[i].imshow(LR.squeeze(0).transpose(0, 2), extent=[0, 1, 0, 1])
        axes[i+cols].imshow(HR.squeeze(0).transpose(0, 2), extent=[0, 1, 0, 1])


    plt.tight_layout()
    plt.show()



n = 50
ds = Dataset_HF.load_from_disk(os.path.join("./datasets_lens", "Lens_2")).select(range(n))
# ds = Dataset_HF.load_from_disk(os.path.join("./datasets_lens", "Lens_2"))


trainset = SuperResolutionDataset_2(ds)


cols = 3

for i in range(int(len(trainset)/cols)):
    plot_imgs(trainset, i*cols, cols)