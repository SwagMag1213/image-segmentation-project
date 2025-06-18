import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from advanced_models import UNetWithBackbone
from dataset import prepare_augmented_data
from utils import get_device

def enable_dropout(model):
    """Activate dropout layers during inference for MC Dropout."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def mc_sigmoid_uncertainty(model, image, T=30):
    """Compute uncertainty via MC Dropout and sigmoid-entropy."""
    model.eval()
    enable_dropout(model)
    sigmoids = []
    with torch.no_grad():
        for _ in range(T):
            logit = model(image.unsqueeze(0))   # [1,1,H,W]
            p = torch.sigmoid(logit)            # [1,1,H,W]
            sigmoids.append(p.cpu())
    probs = torch.stack(sigmoids)            # [T,1,1,H,W]
    mean_p = probs.mean(dim=0).squeeze(0)    # [H,W]
    H = -(mean_p * torch.log(mean_p + 1e-8) +
          (1 - mean_p) * torch.log(1 - mean_p + 1e-8))
    return H.mean().item()

def main():
    # 1) Instantiate model exactly as during training
    model = UNetWithBackbone(
        n_classes=1,
        backbone='resnet50',
        pretrained=True,
        use_attention=False
    )

    # 2) Load checkpoint, but only matching shapes
    checkpoint = torch.load("experiments/unet_baseline_20250611_134913/baseline_model.pth")
    model_dict = model.state_dict()
    filtered = {
        k: v for k, v in checkpoint.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }
    model_dict.update(filtered)
    model.load_state_dict(model_dict)

    device = get_device()
    model.to(device)

    # 3) Prepare data loader (no augmentation)
    train_loader, _ = prepare_augmented_data(
        data_dir='data/manual_labels',
        image_type='W',
        batch_size=1,
        img_size=(128, 128),
        seed=42
    )

    # 4) Compute uncertainties
    uncertainties = []
    for images, _ in tqdm(train_loader, desc="Calculating uncertainties"):
        img = images[0].to(device)
        u = mc_sigmoid_uncertainty(model, img, T=30)
        uncertainties.append(u)

    # 5) Save to .npy
    np.save("uncertainties.npy", np.array(uncertainties))
    print(f"Saved uncertainties for {len(uncertainties)} images to uncertainties.npy")

if __name__ == "__main__":
    main()