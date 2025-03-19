"""
Categories of experiments to test different aspects of the model
"""

# 1. Model Architecture Experiments
def get_backbone_experiments():
    """Test different backbone architectures"""
    return [
        {
            'experiment': 'backbone_resnet34',
            'params': {'backbone': 'resnet34'}
        },
        {
            'experiment': 'backbone_resnet50',
            'params': {'backbone': 'resnet50'}
        },
        {
            'experiment': 'backbone_densenet121',
            'params': {'backbone': 'densenet121'}
        }
    ]

def get_attention_experiments():
    """Test with and without attention mechanisms"""
    return [
        {
            'experiment': 'no_attention',
            'params': {'use_attention': False}
        },
        {
            'experiment': 'with_attention',
            'params': {'use_attention': True}
        }
    ]

# 2. Loss Function Experiments
def get_loss_function_experiments():
    """Test different loss functions"""
    return [
        {
            'experiment': 'loss_combo',
            'params': {'loss_fn': 'combo', 'loss_alpha': 0.5}
        },
        {
            'experiment': 'loss_dice',
            'params': {'loss_fn': 'dice'}
        },
        {
            'experiment': 'loss_focal',
            'params': {'loss_fn': 'focal', 'loss_alpha': 0.25, 'focal_gamma': 2.0}
        },
        {
            'experiment': 'loss_tversky',
            'params': {'loss_fn': 'tversky', 'loss_alpha': 0.3, 'loss_beta': 0.7}
        },
        {
            'experiment': 'loss_boundary',
            'params': {'loss_fn': 'boundary'}
        }
    ]

# 3. Optimizer and Learning Rate Experiments
def get_optimizer_experiments():
    """Test different optimizers and learning rates"""
    return [
        {
            'experiment': 'optimizer_adam',
            'params': {'optimizer': 'adam', 'learning_rate': 1e-4}
        },
        {
            'experiment': 'optimizer_sgd',
            'params': {'optimizer': 'sgd', 'learning_rate': 1e-3}
        },
        {
            'experiment': 'lr_lower',
            'params': {'learning_rate': 5e-5}
        },
        {
            'experiment': 'lr_higher',
            'params': {'learning_rate': 5e-4}
        }
    ]

# 4. Data Processing Experiments
def get_image_size_experiments():
    """Test different image sizes"""
    return [
        {
            'experiment': 'img_size_256',
            'params': {'img_size': (256, 256)}
        },
        {
            'experiment': 'img_size_384',
            'params': {'img_size': (384, 384)}
        },
        {
            'experiment': 'img_size_512',
            'params': {'img_size': (512, 512)}
        }
    ]

# 5. Batch Size Experiments
def get_batch_size_experiments():
    """Test different batch sizes"""
    return [
        {
            'experiment': 'batch_size_2',
            'params': {'batch_size': 2}
        },
        {
            'experiment': 'batch_size_4',
            'params': {'batch_size': 4}
        },
        {
            'experiment': 'batch_size_8',
            'params': {'batch_size': 8}
        }
    ]

# 6. Image Type Experiments
def get_image_type_experiments():
    """Test different image types (B or W)"""
    return [
        {
            'experiment': 'image_type_B',
            'params': {'image_type': 'B'}
        },
        {
            'experiment': 'image_type_W',
            'params': {'image_type': 'W'}
        }
    ]
