# TODO check VIT last layer

import torch
import torch.nn as nn
import timm
 
def create_model(model_name, num_classes, num_input_channels):


    model = timm.create_model(model_name, pretrained=False)

    
    # 1️⃣ modify first convolutional layer for the number of input channels
    if hasattr(model, 'conv_stem'):  # EfficientNet
        model.conv_stem = nn.Conv2d(
            in_channels=num_input_channels, 
            out_channels=model.conv_stem.out_channels,
            kernel_size=model.conv_stem.kernel_size,
            stride=model.conv_stem.stride,
            padding=model.conv_stem.padding,
            bias=model.conv_stem.bias
        )
    elif hasattr(model, 'conv1'):  #ResNet
        model.conv1 = nn.Conv2d(
            in_channels=num_input_channels, 
            out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=model.conv1.bias
        )
    elif hasattr(model, 'patch_embed'):  # Vision Transformer (ViT)
        model.patch_embed.proj = nn.Conv2d(
            in_channels=num_input_channels, 
            out_channels=model.patch_embed.proj.out_channels,
            kernel_size=model.patch_embed.proj.kernel_size,
            stride=model.patch_embed.proj.stride,
            padding=model.patch_embed.proj.padding,
            bias=model.patch_embed.proj.bias
        )
    elif hasattr(model, 'Conv2d_1a_3x3'):  # Inception V3
        model.Conv2d_1a_3x3.conv = nn.Conv2d(
            in_channels=num_input_channels, 
            out_channels=model.Conv2d_1a_3x3.conv.out_channels,
            kernel_size=model.Conv2d_1a_3x3.conv.kernel_size,
            stride=model.Conv2d_1a_3x3.conv.stride,
            padding=model.Conv2d_1a_3x3.conv.padding,
            bias=model.Conv2d_1a_3x3.conv.bias
        )
    else:
        raise ValueError(f"first conv layer - unsupported model architecture: {model_name}")
    


    # 2️⃣ modify the last layer to adopt model for appropriate number of classes
    if hasattr(model, 'classifier'):  # EfficientNet
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'head'):  # RegNet, VIT
        num_features = model.head.fc.in_features
        model.head.fc = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'fc'):  # Inception V3
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'last_linear'):  # Inception V1
        num_features = model.last_linear.in_features
        model.last_linear = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"last layer - unsupported model architecture: {model_name}")
    
    return model