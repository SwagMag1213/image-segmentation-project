import torch
import matplotlib.pyplot as plt
from dataset import load_original_data, CellSegmentationDataset
from tqdm import tqdm
import cv2
import os

def visualize_predictions(model, loader, device, num_samples=3, save_path=None):
    """
    Visualize model predictions vs ground truth (supports U-Net and W-Net)
    
    Args:
        model: Neural network model
        loader: DataLoader containing images and masks
        device: Device to run on (cuda, mps, or cpu)
        num_samples: Number of samples to visualize
        save_path: Optional path to save the figure
    """
    model.eval()
    
    # Get a batch of samples
    images, masks = next(iter(loader))
    
    # Move to device
    images = images.to(device)
    
    # Generate predictions
    with torch.no_grad():
        outputs = model(images)

    # Handle W-Net (tuple output) or U-Net (single output)
    if isinstance(outputs, tuple):
        preds, recon = outputs
    else:
        preds = outputs
        recon = None

    # Move back to CPU for visualization
    images = images.cpu()
    masks = masks.cpu()
    preds = torch.sigmoid(preds).cpu()
    if recon is not None:
        recon = recon.cpu()
    
    # Limit to requested number of samples
    num_samples = min(num_samples, len(images))
    
    # Define number of rows based on model type
    rows = 4 if recon is not None else 3
    fig, axs = plt.subplots(rows, num_samples, figsize=(4*num_samples, 4*rows))
    
    for i in range(num_samples):
        # Display image
        axs[0, i].imshow(images[i, 0], cmap='gray')
        axs[0, i].set_title(f'Input Image')
        axs[0, i].axis('off')
        
        # Display ground truth
        axs[1, i].imshow(masks[i, 0], cmap='gray')
        axs[1, i].set_title(f'Ground Truth')
        axs[1, i].axis('off')
        
        # Display prediction
        axs[2, i].imshow((preds[i, 0] > 0.5).float(), cmap='gray')
        axs[2, i].set_title(f'Prediction')
        axs[2, i].axis('off')

        if recon is not None:
            axs[3, i].imshow(recon[i, 0], cmap='gray')
            axs[3, i].set_title("Reconstruction")
            axs[3, i].axis('off')
    
    plt.tight_layout()
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def plot_broadband_vs_fluorescence(data_dir='manual_labels', save_path='figures/image_comparison.pdf'):
    data_w = load_original_data(data_dir, image_type='W')
    data_b = load_original_data(data_dir, image_type='B')
    
    img_w = cv2.imread(data_w['image_paths'][0], cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(data_b['image_paths'][0], cv2.IMREAD_GRAYSCALE)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_w, cmap='gray')
    axs[0].set_title('Broadband Image (W)')
    axs[0].axis('off')
    
    axs[1].imshow(img_b, cmap='gray')
    axs[1].set_title('Fluorescence Image (B)')
    axs[1].axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved comparison to {save_path}")
    plt.show()

def plot_class_imbalance(image_type='W', data_dir='manual_labels', img_size=(256, 256),
                         save_path='figures/class_imbalance_bar.pdf'):
    data = load_original_data(data_dir, image_type=image_type)
    dataset = CellSegmentationDataset(data['image_paths'], data['mask_paths'], img_size=img_size)

    total_pixels = 0
    foreground_pixels = 0

    for _, mask in tqdm(dataset, desc="Calculating foreground ratio"):
        mask_np = mask.squeeze().numpy()
        foreground_pixels += mask_np.sum()
        total_pixels += mask_np.size

    ratio = foreground_pixels / total_pixels
    bg_ratio = 1 - ratio

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(['Background', 'Foreground'], [bg_ratio * 100, ratio * 100], color=['gray', 'red'])
    ax.set_ylabel('Pixel Percentage (%)')
    ax.set_title('Class Imbalance in Cancer Cell Masks')
    ax.set_ylim(0, 100)
    for i, v in enumerate([bg_ratio * 100, ratio * 100]):
        ax.text(i, v + 1, f"{v:.2f}%", ha='center')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved imbalance plot to {save_path}")
    plt.show()
    
    return ratio


if __name__ == "__main__":
    plot_broadband_vs_fluorescence()
    plot_class_imbalance(image_type='W', data_dir='manual_labels', img_size=(256, 256))