import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from utils import SuperResolutionDataset
from model import WaveMixSR
from datasets import Dataset as Dataset_HF
import os
from tqdm import tqdm



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

resolution = 2


model = WaveMixSR(
    depth = 3,
    mult = 1,
    ff_channel = 144,
    final_dim = 144,
    dropout = 0.3
)

model = model.to(device)


#* without 
path = 'weights/Task3_weights.pth'
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))



ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr = PeakSignalNoiseRatio().to(device)

model.eval()

ds = Dataset_HF.load_from_disk(os.path.join("./datasets_lens", "Lens_2")).select(range(100))

valset = SuperResolutionDataset(ds)


batch_size = 1
testloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                        shuffle=False)

PSNR = 0.
SSIM = 0.

with torch.no_grad():
    with tqdm(testloader, unit="batch") as tepoch:
        for data in tepoch:

            images, labels = data[0].to(device), data[1].to(device) 
            outputs = model(images) 

            PSNR += psnr(outputs, labels) / len(testloader)
            SSIM += ssim(outputs, labels) / len(testloader)


print(f"PSNR: {PSNR:.2f} - SSIM: {SSIM:.4f}\n")