import torch
import random
import os
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

from torchvision import transforms as T
from src.dataset_combine_7_channels import Dataset
from torch.utils.data import DataLoader

def dataset_mean_std(csv, padding):

    padding_transform = T.Compose([
        T.Pad(padding)
    ])
    dataset_padded = Dataset(csv_file=csv, transform=padding_transform)
    loader = DataLoader(dataset_padded, batch_size=64, shuffle=False)

    channel_sum = 0
    channel_squared_sum = 0
    num_pixels = 0

    for images, y_true in loader:
        images = images.view(images.size(0), images.size(1), -1) # torch.Size([64, 7, 50176]) - batch_size, channels, height * width
        
        channel_sum += images.sum(dim=[0, 2])
        channel_squared_sum += (images ** 2).sum(dim=[0, 2])
        num_pixels += images.size(0) * images.size(2)

    mean = channel_sum / num_pixels
    std = ((channel_squared_sum / num_pixels) - (mean ** 2)).sqrt()

    return(mean, std)
