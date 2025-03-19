import torch
import matplotlib.pyplot as plt

def visualize_predictions(model, loader, device, num_samples=3):
    """
    Visualize model predictions vs ground truth
    
    Args:
        model: Neural network model
        loader: DataLoader containing images and masks
        device: Device to run on (cuda, mps, or cpu)
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Get a batch of samples
    images, masks = next(iter(loader))
    
    # Move to device
    images = images.to(device)
    
    # Generate predictions
    with torch.no_grad():
        preds = torch.sigmoid(model(images))
    
    # Move back to CPU for visualization
    images = images.cpu()
    masks = masks.cpu()
    preds = preds.cpu()
    
    # Limit to requested number of samples
    num_samples = min(num_samples, len(images))
    
    # Create figure
    fig, axs = plt.subplots(3, num_samples, figsize=(4*num_samples, 10))
    
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
    
    plt.tight_layout()
    plt.show()