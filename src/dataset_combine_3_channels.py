import os
from PIL import Image
from torch.utils.data import Dataset

class Dataset_Combine_Three_Channels(Dataset):
    def __init__(self, root_dir, transform=None, indices=None):

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # List to store (path, label)
        self.indices = indices

        # Create a list of tuples: [(channel_paths, label), ...]
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for filename in os.listdir(class_path):
                if filename.endswith("_ch1.png"):  # for each "*_ch1.png" file we extract base_name
                    base_name = filename[:-8]  # Remove "_ch1.png"
                    sample = (
                        [
                            os.path.join(class_path, f"{base_name}_{c}.png")
                            for c in ["ch1", "ch5", "ch6"]
                        ],
                        class_idx,
                    )
                    self.samples.append(sample)

        # Initialize indices
        if indices is None:
            # If no indices are provided, use all indices
            self.indices = list(range(len(self.samples)))
        else:
            # Validate and store provided indices
            if max(indices) >= len(self.samples):
                raise ValueError("Provided indices exceed dataset size.")
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map the subset index to the original sample index
        original_idx = self.indices[idx]
        channel_paths, label = self.samples[original_idx]

        # Open and merge the channels
        channels = [Image.open(path) for path in channel_paths]
        image = Image.merge("RGB", channels)  # Merge channels into an RGB image

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
