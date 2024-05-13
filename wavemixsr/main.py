import torch
import torch.nn.functional as F


LR = torch.rand((1,1,64,64))

# Upsample LR using nearest interpolation with scale_factor=2
LR = F.interpolate(LR, scale_factor=2, mode="nearest")

print(LR.shape)