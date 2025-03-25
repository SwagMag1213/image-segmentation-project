import torch
import matplotlib.pyplot as plt

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