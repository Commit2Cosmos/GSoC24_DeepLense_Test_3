
import torch
import torch.mps
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import time
from torch.utils.data import Dataset
import torchvision.transforms as transforms
# from torchinfo import summary
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
import numpy as np
from wavemix import Level1Waveblock
import os

from utils import train_test_eval_split
from datasets import Dataset as Dataset_HF

if __name__ == '__main__':
    
    device = torch.device("mps")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # torch.backends.cudnn.benchmarks = True
    # torch.backends.cudnn.deterministic = True


    resolution = 2
    #* ssim or psnr
    eval_metric = "psnr"
    test_dataset_name = "lens"

    n = 20

    ds = Dataset_HF.load_from_disk(os.path.join("./datasets_lens", "Lens")).select(range(n))

    dataset_train, dataset_test, dataset_val = train_test_eval_split(ds)


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


    trainset = SuperResolutionDataset(dataset_train)
    valset = SuperResolutionDataset(dataset_val)
    testset = SuperResolutionDataset(dataset_test)


    print(len(trainset))
    print(len(valset))
    print(len(testset))


    class WaveMixSR(nn.Module):
        def __init__(
            self,
            *,
            depth,
            mult = 1,
            ff_channel = 16,
            final_dim = 16,
            dropout = 0.3,
        ):
            super().__init__()
            
            self.layers = nn.ModuleList([])
            for _ in range(depth):
                self.layers.append(Level1Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
            
            self.final = nn.Sequential(
                nn.Conv2d(final_dim, int(final_dim/2), 3, stride=1, padding=1),
                nn.Conv2d(int(final_dim/2), 1, 1)
            )


            self.path1 = nn.Sequential(
                nn.Upsample(scale_factor=int(resolution), mode='bilinear', align_corners = False),
                nn.Conv2d(1, int(final_dim/2), 3, 1, 1),
                nn.Conv2d(int(final_dim/2), final_dim, 3, 1, 1)
            )

        def forward(self, img):

            img = self.path1(img)

            for attn in self.layers:
                img = attn(img) + img

            img = self.final(img)

            return img


    model = WaveMixSR(
        depth = 2,
        mult = 1,
        ff_channel = 144,
        final_dim = 144,
        dropout = 0.3
    )

    model = model.to(device)

    batch_size = 2

    PATH = str(test_dataset_name)+'_'+str(resolution)+'_'+str(eval_metric)+'.pth'

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False)


    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    criterion = nn.HuberLoss() 

    # scaler = torch.cuda.amp.GradScaler()
    toppsnr = []
    topssim = []
    traintime = []
    testtime = []
    counter = 0


    #* Fist optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    epoch = 0
    while counter < 25:
        
        t0 = time.time()
        epoch_psnr = 0
        epoch_loss = 0
        
        model.train()

        with tqdm(trainloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")

            for data in tepoch:
        
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                # outputs = outputs[:, 0:1, :, :]
                # labels = labels[:, 0:1, :, :]
            
                # with torch.cuda.amp.autocast():
                loss = criterion(outputs, labels) 
                loss.backward()
                optimizer.step()
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

                epoch_PSNR = psnr(outputs, labels) 
                epoch_SSIM = structural_similarity_index_measure(outputs, labels)
                
                epoch_loss += loss / len(trainloader)
                tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - PSNR: {epoch_PSNR:.4f} - SSIM: {epoch_SSIM:.4f}" )

        model.eval()
        t1 = time.time()
        PSNR = 0
        sim = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)

                # outputs = outputs[:, 0:1, :, :]
                # labels = labels[:, 0:1, :, :]

                PSNR += psnr(outputs, labels) / len(testloader)
                sim += structural_similarity_index_measure(outputs, labels) / len(testloader)

        
        print(f"Epoch : {epoch+1} - PSNR_y: {PSNR:.2f} - SSIM_y: {sim:.4f}  - Test Time: {time.time() - t1:.0f}\n")

        topssim.append(sim)
        toppsnr.append(PSNR)
        traintime.append(t1 - t0)
        testtime.append(time.time() - t1)
        counter += 1
        epoch += 1

        if eval_metric == 'psnr':
            if float(PSNR) >= float(max(toppsnr)):
                torch.save(model.state_dict(), PATH)
                print(1)
                counter = 0

        else:
            if float(sim) >= float(max(topssim)):
                torch.save(model.state_dict(), PATH)
                print(1)
                counter = 0

    counter  = 0
    model.load_state_dict(torch.load(PATH))
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #Second Optimizer

    while counter < 25:  # loop over the dataset multiple times
        t0 = time.time()
        epoch_psnr = 0
        epoch_loss = 0
        running_loss = 0.0
        
        model.train()

        with tqdm(trainloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")

            for data in tepoch:
        
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
            #   outputs = outputs[:, 0:1, :, :]
            #   labels = labels[:, 0:1, :, :]

            #   with torch.cuda.amp.autocast():
                loss = criterion(outputs, labels) 
                loss.backward()
                optimizer.step()

                epoch_PSNR = psnr(outputs, labels) 
                epoch_SSIM = structural_similarity_index_measure(outputs, labels)
                epoch_loss += loss / len(trainloader)
                tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - PSNR: {epoch_PSNR:.4f} - SSIM: {epoch_SSIM:.4f}" )

        t1 = time.time()
        model.eval()
        PSNR = 0   
        sim = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)

                # outputs = outputs[:, 0:1, :, :]
                # labels = labels[:, 0:1, :, :]

                PSNR += psnr(outputs, labels) / len(testloader)
                sim += structural_similarity_index_measure(outputs, labels) / len(testloader)

        
        print(f"Epoch : {epoch+1} - PSNR_y: {PSNR:.2f} - SSIM_y: {sim:.4f}  - Test Time: {time.time() - t1:.0f}\n")

        topssim.append(sim)
        toppsnr.append(PSNR)

        traintime.append(t1 - t0)
        testtime.append(time.time() - t1)
        epoch += 1
        counter += 1

        if eval_metric == 'psnr':
            if float(PSNR) >= float(max(toppsnr)):
                torch.save(model.state_dict(), PATH)
                print(1)
                counter = 0

        else:
            if float(sim) >= float(max(topssim)):
                torch.save(model.state_dict(), PATH)
                print(1)
                counter = 0

    print('Finished Training')
    model.load_state_dict(torch.load(PATH))

    print('Results for Div2k val')
    print(f"PSNR_y: {float(max(toppsnr)):.4f} - SSIM_y: {float(max(topssim)):.4f}  - Test Time: {min(testtime):.0f}\n")


    PSNR_y = 0
    SSIM_y = 0

    with torch.no_grad():
        for data in testloader:

            images, labels = data[0].to(device), data[1].to(device) 
            outputs = model(images) 

            # Extract Y Channel
            outputs_ychannel = outputs[:, 0:1, :, :]
            labels_ychannel = labels[:, 0:1, :, :]

            
            PSNR_y += psnr(outputs_ychannel, labels_ychannel) / len(testloader)
            SSIM_y += float(ssim(outputs_ychannel, labels_ychannel)) / len(testloader)


    print("Training and Validating in Div2k")
    print(f"Dataset\n")
    print(str(test_dataset_name))
    print(f"Train Time: {min(traintime):.0f} -Test Time: {min(testtime):.0f}\n")
    print("In Y Space")
    print('evaluation Metric')
    print(str(eval_metric))
    print(f"PSNR: {PSNR_y:.2f} - SSIM: {SSIM_y:.4f}\n")