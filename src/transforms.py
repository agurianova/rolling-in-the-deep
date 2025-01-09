from torchvision.transforms import Compose, Pad, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter

def create_transform(padding=None, fill=0, augmentations=None,mean=None, std=None):

    transforms = []
    
    # padding
    if padding:
        transforms.append(Pad(padding=padding, fill=fill))
    
    # augmentation
    if augmentations:
        transforms.extend(augmentations)
    
    # ToTensor
    transforms.append(ToTensor())
    
    # normalization
    if mean and std:
        transforms.append(Normalize(mean=mean, std=std))
    
    return Compose(transforms)
