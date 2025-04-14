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
    elif "convnext" in model_name.lower(): # ConvNext
        first_conv = model.stem[0]
        model.stem[0] = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
    elif hasattr(model, 'stem'):  # RegNet
        model.stem.conv = nn.Conv2d(
            in_channels=num_input_channels, 
            out_channels=model.stem.conv.out_channels,
            kernel_size=model.stem.conv.kernel_size,
            stride=model.stem.conv.stride,
            padding=model.stem.conv.padding,
            bias=model.stem.conv.bias
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
    elif hasattr(model, 'conv1'):  # ResNet / ResNeXt
        first_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
    else:
        raise ValueError(f"first conv layer - unsupported model architecture: {model_name}")
    


    # 2️⃣ modify the last layer to adopt model for appropriate number of classes
    if hasattr(model, 'classifier'):  # EfficientNet
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'head'):
        if hasattr(model.head, 'fc'):
            num_features = model.head.fc.in_features
            model.head.fc = nn.Linear(num_features, num_classes)
        elif isinstance(model.head, nn.Linear):
            num_features = model.head.in_features
            model.head = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'head'):  # RegNet (comment)
        num_features = model.head.fc.in_features
        model.head.fc = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'fc'):  # Inception V3 / ResNet / ResNeXt
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"last layer - unsupported model architecture: {model_name}")
    
    return model 