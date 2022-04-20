import torch
import random

import numpy as np
import torch.backends.cudnn as cudnn


def gener_noise(gener_batch_size, latent_dim):
    return torch.FloatTensor(np.random.normal(0, 1, (gener_batch_size, latent_dim)))

def calculate_gradient_penalty(model, real_images, fake_images, constant=1.0, lamb=10.0):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.Tensor(np.random.random((real_images.size(0), 1, 1, 1))).to(real_images.get_device())
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones([real_images.shape[0], 1], requires_grad=False).to(real_images.get_device())

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lamb
    
    return gradient_penalty


# wgan_loss
def wgan_loss(pred, real_or_not=True):
    if real_or_not:
        return - torch.mean(pred)
    else:
        return torch.mean(pred)


# Source from "https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py"
def init_torch_seeds(seed: int = 0):
    r""" Sets the seed for generating random numbers. Returns a
    Args:
        seed (int): The desired seed.
    """
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    print("Initialize random seed.")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
