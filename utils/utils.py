import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def psnr_np(input: np.ndarray, target: np.ndarray):
    assert input.shape == target.shape, "no matching shape"
    return -10 * np.log10(np.mean((input - target) ** 2.))

def mkdir(dir_name: str):
    os.makedirs(dir_name, exist_ok=True)

def visualize(epoch, img, filepath, mode, filename=None, video_writer=None):
    outdir = os.path.join(filepath, mode)
    mkdir(outdir)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img * 255).astype(np.uint8)
    if mode == "val":
        cv2.imwrite(os.path.join(outdir, filename), img)
    else:
        video_writer.write(img)
    # plt.title(epoch)
    # plt.axis("off")
    # plt.imshow(img, cmap="gray")
    # plt.savefig(os.path.join(filepath, filename))
    # plt.imsave(os.path.join(filepath, filename), img)


def load_ckpt(
        filename: str,
):
    checkpoint = torch.load(filename)
    return checkpoint
    # if "model_fine_state_dict" in checkpoint:
    #     return checkpoint["epoch"], \
    #             checkpoint["best_psnr"], \
    #             checkpoint["model_state_dict"], \
    #             checkpoint["optimizer_state_dict"], \
    #             checkpoint["losslist"], \
    #             checkpoint["model_fine_state_dict"]
    # return checkpoint["epoch"], \
    #         checkpoint["best_psnr"], \
    #         checkpoint["model_state_dict"], \
    #         checkpoint["optimizer_state_dict"], \
    #         checkpoint["losslist"], \
    #         None

def save_ckpt(
        filename: str,
        args,
        epoch: int,
        best_psnr: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler = None,
        losslist: list = None,
        model_fine: torch.nn.Module = None,
):
    checkpoint = {
        "args": args,
        "epoch": epoch,
        "best_psnr": best_psnr,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "losslist": losslist,
        "model_fine_state_dict": None if model_fine is None else model_fine.state_dict(),
    }

    if filename[-4:] != ".pth":
        filename += ".pth"

    torch.save(checkpoint, filename)

def posenc(L, x):
    res = [x]
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            res.append(fn(2.**i * x))
    res = torch.concat(res, -1)
    return res

def batchify(f, inputs, viewdirs=None, chunk=1024*32, use_dirs=True):
    rgblist, sigmalist = [], []
    for i in range(0, inputs.shape[0], chunk):
        if use_dirs:
            rgb, sigma = f(inputs[i:i+chunk], viewdirs[i:i+chunk])
        else:
            rgb, sigma = f(inputs[i:i+chunk])
        rgblist.append(rgb)
        sigmalist.append(sigma)
    rgb = torch.concat(rgblist, 0)
    sigma = torch.concat(sigmalist, 0)
    return rgb, sigma

def sample_pdf(z_vals: torch.Tensor, weights: torch.Tensor, N_samples: int, training: bool, det: bool=False, device=None):
    
    if device is None:
        device = 'cpu'
    # prevent nan
    weights = weights + 1e-5
    weights = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(weights, -1) # N_rays x N_sample
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    new_shape = list(cdf.shape[:-1]) + [N_samples] # N_sample = N_importance

    eps = 1e-5

    if det:
        u = torch.linspace(0.+eps, 1.-eps, N_samples, device=device).expand(new_shape)
    else:
        u = torch.rand(new_shape).to(device)
    u = u * (1-2*eps) + eps

    # print(u)

    if not training:
        np.random.seed(0)
        u = np.linspace(0.+eps, 1.-eps, N_samples)
        u = np.broadcast_to(u, new_shape)
        if not det:
            u = np.random.rand(*new_shape) * (1 - 2*eps) + eps
        u = torch.Tensor(u).to(device=device)

    u = u.contiguous()
    idx = torch.searchsorted(cdf, u, right=True)
    low = torch.max(torch.zeros_like(idx), idx-1)
    up = torch.min(torch.ones_like(idx)*z_vals.shape[-1] - 1, idx)

    leftbound = torch.gather(z_vals, -1, low)
    rightboud = torch.gather(z_vals, -1, up)
    # print(f"left: {leftbound} \n right: {rightboud}")
    cdf_up = torch.gather(cdf, -1, up)
    cdf_low = torch.gather(cdf, -1, low)
    denom = cdf_up - cdf_low
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    u = (u - cdf_low) / denom
    u = leftbound + u * (rightboud - leftbound)

    return u

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d
    
# test
# import sys
# sys.path.append('/home/x/xie77777/codes/mynerf')
# from model.NeRF import NeRF

# print(sample_pdf(torch.Tensor([[1, 2, 3]]), torch.Tensor([[0.3, .5, .2]]), 6, True))
# psnr_np(np.random.randn(3, 3, 3), np.random.randn(3, 3))