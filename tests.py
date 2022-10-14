import torch
import numpy as np

from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5)),
])

x = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
print(x[1])
