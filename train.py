'''
generate.py

Copyright (c) 2021 Jude Davis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import sys
import torch
import contextlib

from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from model import *
from dataset import GenericDataset, load_data, WIDTH, HEIGHT


# - Constants
G_LR = .0002
D_LR = .0002
BETA_1 = 0.5
BATCH_SIZE = 128
DATASET_FOLDER = 'data'
FROM_PRETRAINED = False
N_OUTPUT_IMGS = BATCH_SIZE
IMG_SIZE = WIDTH * HEIGHT * CHANNELS
N_TRAINING_IMGS = len(os.listdir(DATASET_FOLDER))
EPOCHS = int(sys.argv[1]) if __name__ == '__main__' else None

cpu_dev = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.backends.mps.is_built(): device = torch.device('mps')

print(N_TRAINING_IMGS)
print(str(device).upper())


def main():
    data    = load_data(DATASET_FOLDER)
    dataset = GenericDataset(data)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    gen = G().to(device)  # Generator
    disc = D().to(device)  # Discriminator
    gen.apply(weights_init)
    disc.apply(weights_init)

    if FROM_PRETRAINED:
        gen.load('Generator.model')
        disc.load('Discriminator.model')
    gen.train()
    disc.train()

    gen = gen.to(device)
    disc = disc.to(device)

    crit = nn.BCELoss()
    G_optimizer = torch.optim.AdamW(gen.parameters(), lr=G_LR, betas=(BETA_1, 0.999))
    D_optimizer = torch.optim.AdamW(disc.parameters(), lr=D_LR, betas=(BETA_1, 0.999))
    
    t_steps = len(loader)
    
    clear()
    print(f'\nNumber of steps/epoch: {str(t_steps)}\n')

    # one-line helper functions
    generate_label = lambda x: torch.ones(x.size(0)).to(device)
    grid_permute = lambda x: make_grid(x.detach()[-1], normalize=True).permute(1, 2, 0)

    # - introducing label smoothing
    real_ = 1
    fake_ = 0

    loss_history = []
    t = tqdm(range(EPOCHS))

    with (
        torch.autocast(gen.device) if gen.device == 'cuda'
        else contextlib.nullcontext()
    ):
        for i in t:
            for j, img in enumerate(loader):
                # - train discriminator
                disc.zero_grad()

                decision: torch.Tensor = disc(img.to(device)).view(-1)
                D_loss_real = crit(decision, generate_label(decision).fill_(real_))

                fake_imgs = gen(generate_noise())
                decision = disc(fake_imgs.detach())
                D_loss_fake = crit(decision.flatten(), generate_label(decision).fill_(fake_))

                D_loss: torch.Tensor = (D_loss_fake + D_loss_real)
                D_loss.backward()
                D_optimizer.step()

                # - train generator
                gen.zero_grad()
                decision = disc(fake_imgs)

                G_loss = crit(decision.flatten(), generate_label(decision).fill_(1))

                G_loss.backward()
                G_optimizer.step()
                
                # t.set_description(f'Batch: {j + 1}/{t_steps} G-Loss: {G_loss.item():.5f}')
                t.set_description(f'Batch: {j + 1}/{t_steps} G-Loss: {G_loss.item():.5f} D-Loss: {D_loss.item():.5f}')
            
            loss_history.append(G_loss.item())
    
    plt.plot(range(EPOCHS), loss_history)
    plt.savefig('loss-history.png')
    plt.show()

    gen.save('Generator.model')
    disc.save('Discriminator.model')
    
    gen.eval()
    image: torch.Tensor = gen(generate_noise())
    test_loss = crit(decision, generate_label(decision).fill_(1)).detach()

    print(f'Testing Loss: {test_loss}')

    img_grid = grid_permute(image.cpu())
    plt.imshow(img_grid.numpy())
    plt.show()

    plt.imsave('image.png', img_grid.numpy())


    print('Saved!')


def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def generate_noise() -> torch.Tensor:
    return torch.randn(N_OUTPUT_IMGS, LATENT_SPACE_SIZE, 1, 1, device=str(device)).float()

def clear():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


if __name__ == '__main__':
    main()
