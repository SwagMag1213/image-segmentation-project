import torch
from losses import SoftNCutLoss

# Create dummy grayscale images (B=2, C=1, H=256, W=256)
images = torch.rand(2, 1, 256, 256)

# Create dummy segmentation output (softmax style)
seg_output = torch.rand(2, 3, 256, 256)  # 3 = number of soft segments

# Initialize the loss
loss_fn = SoftNCutLoss(img_size=(256, 256))

# Forward pass
loss = loss_fn(images, seg_output)

print("SoftNCut Loss:", loss.item())

seg_output.requires_grad = True
loss = loss_fn(images, seg_output)
loss.backward()
print("Backward pass succeeded.")
