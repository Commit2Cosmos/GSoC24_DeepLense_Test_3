
import torch
import torch.mps
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import time
# from torchinfo import summary
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from model import WaveMixSR
import os

from utils import train_test_eval_split, SuperResolutionDataset
from datasets import Dataset as Dataset_HF

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True

    resolution = 2
    #* ssim or psnr
    eval_metric = "psnr"
    test_dataset_name = "lens"

    # n = 200

    # ds = Dataset_HF.load_from_disk(os.path.join("./datasets_lens", "Lens")).select(range(n))
    ds = Dataset_HF.load_from_disk(os.path.join("./datasets_lens", "Lens"))

    dataset_train, dataset_test, dataset_val = train_test_eval_split(ds)


    trainset = SuperResolutionDataset(dataset_train)
    valset = SuperResolutionDataset(dataset_val)
    testset = SuperResolutionDataset(dataset_test)


    print(len(trainset))
    print(len(valset))
    print(len(testset))


    model = WaveMixSR(
        depth = 3,
        mult = 1,
        ff_channel = 144,
        final_dim = 144,
        dropout = 0.3
    )

    model = model.to(device)

    batch_size = 1

    PATH = str(test_dataset_name)+'_'+str(resolution)+'_'+str(eval_metric)+'.pth'

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)


    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    criterion = nn.HuberLoss() 

    scaler = torch.cuda.amp.GradScaler()
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

                with torch.cuda.amp.autocast():
                    loss = criterion(outputs, labels) 
                # loss.backward()
                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_PSNR = psnr(outputs, labels) 
                epoch_SSIM = ssim(outputs, labels)
                
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

                PSNR += psnr(outputs, labels) / len(testloader)
                sim += ssim(outputs, labels) / len(testloader)

        
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

                with torch.cuda.amp.autocast():
                    loss = criterion(outputs, labels) 
                # loss.backward()
                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_PSNR = psnr(outputs, labels) 
                epoch_SSIM = ssim(outputs, labels)
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

                PSNR += psnr(outputs, labels) / len(testloader)
                sim += ssim(outputs, labels) / len(testloader)

        
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

    print('Results for Lens val')
    print(f"PSNR_y: {float(max(toppsnr)):.4f} - SSIM_y: {float(max(topssim)):.4f}  - Test Time: {min(testtime):.0f}\n")


    PSNR_y = 0
    SSIM_y = 0

    with torch.no_grad():
        for data in testloader:

            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            PSNR_y += psnr(outputs, labels) / len(testloader)
            SSIM_y += float(ssim(outputs, labels)) / len(testloader)


    print("Training and Validating in Div2k")
    print(f"Dataset\n")
    print(str(test_dataset_name))
    print(f"Train Time: {min(traintime):.0f} -Test Time: {min(testtime):.0f}\n")
    print("In Y Space")
    print('evaluation Metric')
    print(str(eval_metric))
    print(f"PSNR: {PSNR_y:.2f} - SSIM: {SSIM_y:.4f}\n")