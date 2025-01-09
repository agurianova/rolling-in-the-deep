import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np  

class Dataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file) # load the CSV file 
        self.transform = transform


    def __len__(self):
        return len(self.data)  # number of rows in the CSV (images)
    

    def __getitem__(self, idx):
        """
        Returns:
            Tuple: (image, label) where `image` is a tensor of shape (7, H, W) and `label` is the corresponding label.
        """
       
        image_paths = self.data.iloc[idx, 0:7].tolist()  # extract the paths for all 7 channels
        image = self.load_image(image_paths) # load image using paths

        label = self.data.iloc[idx, 7]  # label is stored in the last column of the CSV

        if self.transform:
            image = self.transform(image)

        return image, label


    def load_image(self, channel_paths):
        """
        Returns:
            torch.Tensor: A tensor of shape (7, H, W) representing the stacked image.
        """
        channels = []  # list of tensors
        for path in channel_paths:
            channel_image = Image.open(path).convert('L') # PIL
            channel_image = torch.tensor(np.array(channel_image), dtype=torch.float32) # PIL >> np array >> tensor
            channels.append(channel_image) # to the list of tensors

        image = torch.stack(channels, dim=0) # shape (7, H, W)

        return image  