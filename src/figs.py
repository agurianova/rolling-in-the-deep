import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image
import io
import torchvision.transforms as T

def fig_confusion_matrix(y_true, y_pred):
    '''
    y_true: np.array
    y_pred: np.array
    '''
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("confusion matrix")

    return fig

def fig_image_details(image, true, prob):
    """
    Parameters:
        image (np.array): Input image array.
        true (int): True class index.
        prob (np.array): Class probabilities.

    Returns:
        torch.Tensor: Image as PyTorch tensor.
    """
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    from torchvision.transforms import ToTensor

    fig, axes = plt.subplots(1, image.shape[0] + 1, figsize=(32, 4))

    # Titles for channels
    titles = [
        "ch1 read_base",
        "ch2 base_quality",
        "ch3 mapping_quality",
        "ch4 strand",
        "ch5 read_supports_variant",
        "ch6 base_differs_from_ref",
        "ch7 haplotype"
    ]

    # Handle image channels
    for i in range(image.shape[0]):
        axes[i].imshow(image[i], cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(titles[i])
        axes[i].axvline(x=111, ymin=162 / 224, ymax=1, color="white", linewidth=1)

    # Plot probabilities
    bar_colors = ["gray"] * len(prob)
    bar_colors[true] = "green"
    axes[image.shape[0]].bar(range(len(prob)), prob, color=bar_colors, edgecolor="white")
    axes[image.shape[0]].set_ylim(0, 1)
    axes[image.shape[0]].set_title("class probabilities")

    plt.tight_layout()

    # Save figure to buffer
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        pil_image = Image.open(buf).convert("RGB")
    finally:
        buf.close()
        plt.close(fig)  # Ensure figure is closed

    # Convert PIL image to PyTorch tensor
    tensor_image = ToTensor()(pil_image)  # (C, H, W)

    return tensor_image
