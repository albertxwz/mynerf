import torch
import torch.nn.functional as F

def criterion(img: torch.Tensor, target: torch.Tensor):
    return F.mse_loss(img, target)

def psnr(img: torch.Tensor, target: torch.Tensor):
    return -10 * torch.log10(F.mse_loss(img, target))