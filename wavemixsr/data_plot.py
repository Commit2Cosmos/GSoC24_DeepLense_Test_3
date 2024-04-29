from datasets import Dataset as Dataset_HF
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from utils import SuperResolutionDataset


def plot_imgs(dataset: SuperResolutionDataset):
    fig, axes = plt.subplots(2, 4, sharex='all', sharey='all', figsize=(15,12))
    plt.axis('off')

    axes = axes.flatten()

    for i in range(0, int(len(axes)/2)):
        axes[i].imshow(trainset[i+10][0].transpose(0, 2), extent=[0, 1, 0, 1])
        axes[i+4].imshow(trainset[i+10][1].transpose(0, 2), extent=[0, 1, 0, 1])


    plt.tight_layout()
    plt.show()


n = 100

# ds = Dataset_HF.load_from_disk(os.path.join("./datasets_lens", "Lens_2")).select(range(n))
ds = Dataset_HF.load_from_disk(os.path.join("./datasets_lens", "Lens_2"))


trainset = SuperResolutionDataset(ds)

plot_imgs(trainset)