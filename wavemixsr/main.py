from model import WaveMixSR


model = WaveMixSR(
    depth = 3,
    mult = 1,
    ff_channel = 144,
    final_dim = 144,
    dropout = 0.3
)

for name, param in model.named_parameters():
    if 'layers' in name or 'path1' in name:
        param.requires_grad = False

for name, param in model.named_parameters():
    print(name, param.requires_grad)