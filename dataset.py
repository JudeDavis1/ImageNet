import os
import cv2

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset



WIDTH = 200
HEIGHT = 200
MAX_IMGS = 841

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((WIDTH, HEIGHT)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.5, 0.7))
])

def load_data(folder_path: str) -> list:
    imgs = []
    filenames = os.listdir(folder_path)

    i = 0
    for i in ( t := tqdm(range(len(filenames))) ):
        img = cv2.imread(os.path.join(folder_path, filenames[i]), cv2.IMREAD_COLOR)
        
        try:
            imgs.append(transform(img))
        except cv2.error:
            continue

        if i % 100 == 0:
            t.set_description(f'Processing image: {i}')

    
    return imgs


class GenericDataset(Dataset):
    
    def __init__(self, dataset):
        self.data = dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
