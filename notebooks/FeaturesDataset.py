from torch.utils.data import Dataset
import numpy as np
from PIL import Image

WIDTH = 128
HEIGHT = WIDTH // 2

def load_img(path):
    image = Image.open(path)
    image = image.resize((WIDTH, HEIGHT))

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    return image


class FeaturesDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        feature = load_img(path)
        if self.transform:
            feature = self.transform(feature)
            
        label = self.labels[idx]
        
        return feature, label