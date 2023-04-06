import torch
import torch.nn as nn

from typing import List


# - Hyperparams
n_featuremap_G = 64
n_featuremap_D = 64

CHANNELS = 3
LATENT_SPACE_SIZE = 100



class BaseModel(nn.Module):
    def save(self, path: str):
        torch.save(self.state_dict(), path)


    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

class G(BaseModel):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            *self.reg_block(LATENT_SPACE_SIZE, n_featuremap_G * 8, 1, 0),
            *self.reg_block(n_featuremap_G * 8, n_featuremap_G * 4),
            *self.reg_block(n_featuremap_G * 4, n_featuremap_G * 2),
            *self.reg_block(n_featuremap_G * 2, n_featuremap_G),
            nn.ConvTranspose2d(n_featuremap_G, CHANNELS, stride=2, kernel_size=4, bias=False),
            nn.Tanh()
        )
    
    def reg_block(self, prev, nfg, stride=2, padding=1, batch_norm=True):
        layers = [
            nn.ConvTranspose2d(prev, nfg, stride=stride, kernel_size=4, padding=padding, bias=False),
            nn.ReLU(True),
        ]

        if batch_norm:
            layers.insert(1, nn.BatchNorm2d(nfg))
        
        return layers
    
    def forward(self, x) -> torch.Tensor:
        return self.model(x)

class D(BaseModel):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            *self.reg_block(CHANNELS, n_featuremap_D, False),
            *self.reg_block(n_featuremap_D, n_featuremap_D * 2),
            *self.reg_block(n_featuremap_D * 2, n_featuremap_D * 4),
            *self.reg_block(n_featuremap_D * 4, n_featuremap_D * 8),
            nn.Conv2d(n_featuremap_D * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def reg_block(self, prev, nfd, batch_norm=True) -> List[torch.Tensor]:
        layers = [
            nn.Conv2d(prev, nfd, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if batch_norm:
            layers.insert(1, nn.BatchNorm2d(nfd))
        
        return layers
    
    def forward(self, x) -> torch.Tensor:
        return self.model(x)