from wavemix import Level1Waveblock
import torch.nn as nn
from torchinfo import summary
import torch
import torch.backends


resolution = 2

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

device = torch.device('mps')

model = WaveMixSR(
    depth = 1,
    mult = 1,
    ff_channel = 144,
    final_dim = 144,
    dropout = 0.3
)



mult = 2
ff_channel = 16
final_dim = 16
dropout = 0.5


model = model.to(torch.device(device))



t = torch.rand((1, 1, 75, 75), device=device)

model(t)

# summary(model, input_size=(1, 1, 75, 75), col_names= ("input_size","output_size","num_params","mult_adds"), depth = 4)



# lin = nn.Linear(10, 10, device=device)

# print(lin(torch.rand((2,10), device=device)))