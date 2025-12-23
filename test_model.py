import torch
from model import OCRNet

model = OCRNet()

# fake batch
x = torch.randn(4, 1, 32, 128)

y = model(x)

print("Output shape:", y.shape)
