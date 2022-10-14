import torch
import numpy as np

from torch import nn

from train import *
from model import G, D
from dataset import load_data



def eval():
    gen = G()
    gen.load('Generator.model')

    disc = D()
    disc.load('Discriminator.model')

    criterion = nn.BCELoss()
    real_imgs = torch.stack(get_testing_images())

    z = torch.randn(BATCH_SIZE, LATENT_SPACE_SIZE, 1, 1)
    fake = gen(z)
    realism = disc(fake).view(-1)
    print(realism.mean().detach())

    print(fake.shape)

    plt.imshow(make_grid(fake).permute(1, 2, 0).detach() * 255)
    plt.show()



def get_testing_images() -> list:
    imgs = load_data(DATASET_FOLDER)
    return imgs[:BATCH_SIZE]


if __name__ == '__main__':
    eval()


