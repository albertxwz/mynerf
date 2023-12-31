# import sys
# sys.path.append("/home/x/xie77777/codes/mynerf")
# print(sys.path)
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
import logging
# device = "cpu"
from utils.utils import batchify, posenc, sample_pdf

def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing='xy',
    )
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i, dtype=np.float32)], -1)
    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)
    return rays_o, rays_d

def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing='xy',
    )
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1).to(dtype=dtype)
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape).to(dtype=dtype)
    return rays_o, rays_d

def volume_render(
    rays_d: torch.Tensor,
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    pts: torch.Tensor,
    z_vals: torch.Tensor,
    white_bkgd: bool = False,
    raw_noise_std: float = 0.,
    testing: bool = False,
):
    # print("sigma, pts:", sigma.shape, pts.shape)
    # dists = torch.concat(
    #     [torch.sum((pts[..., 1:, :] - pts[..., :-1, :])**2., -1)**.5, torch.Tensor([1e10]).to(device).expand(list(pts.shape[:-2]) + [1])],
    #     -1,
    # )
    dists = torch.concat(
        [z_vals[..., 1:] - z_vals[..., :-1], torch.ones_like(z_vals[..., -1:]) * 1e10],
        -1,
    )

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    # logging.info(f"dists: {dists}\nsigma: {sigma}")

    # print(f"dist shape: {dists.shape}")

    noise = 0.
    if raw_noise_std > 0. and not testing:
        noise = torch.randn_like(sigma) * raw_noise_std

        # if testing:
        #     np.random.seed(0)
        #     noise = np.random.randn(*list(sigma.shape)) * raw_noise_std
        #     noise = torch.Tensor(noise).to(device)
    
    sigma = torch.relu(sigma + noise)

    alpha = 1. - torch.exp(-sigma * dists)

    acc = torch.cumprod(1.-alpha + 1e-10, -1)
    acc = torch.concat([torch.ones_like(alpha[..., :1]), acc[..., :-1]], -1)
    weights = alpha * acc
    # logging.info(f"alpha {alpha}\nacc {acc}")

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    # print("weight, zvals, rgbmap", weights.shape, z_vals.shape, rgb_map.shape)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights

def render_rays(
    args,
    model: torch.nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    N_samples: int,
    perturb: bool = True,
    # fine-grained net
    model_fine: torch.nn.Module = None,
    N_importance: int = 0,
    dirs = None,
):
    # mine not correct ?!
    # z_vals = torch.linspace(near, far, N_samples+1, device=device)[..., :-1] + (far - near) / N_samples * .5

    N_rays = rays_o.shape[0]

    z_vals = torch.linspace(near, far, N_samples, device=device)

    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [N_samples])

    if perturb:

        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.concat([mids, z_vals[..., -1:]], -1)
        lower = torch.concat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand_like(z_vals)

        if not model.training:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand).to(device)
        
        z_vals = lower + (upper - lower) * t_rand

        # new_shape = list(rays_o.shape[:-1]) + [N_samples]
        # if model.training:
        #     z_vals = z_vals + (torch.rand(new_shape, device=device) - .5) * (far - near) / N_samples
        # else:
        #     np.random.seed(0)
        #     z_vals = z_vals + (torch.Tensor(np.random.rand(*new_shape) - .5).to(device) * (far - near) / N_samples)
    # z_vals = z_vals.double()

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]

    inputs = pts.reshape([-1, 3])
    inputs = posenc(args.L, inputs)
    if args.use_dirs:
        viewdirs = dirs[..., None, :].expand_as(pts).reshape([-1, 3])
        viewdirs = posenc(args.L_dirs, viewdirs)
        rgb, sigma = batchify(model, inputs, viewdirs, use_dirs=True)
        # rgb, sigma = model(inputs, viewdirs)
    else:
        rgb, sigma = batchify(model, inputs, use_dirs=False)
        # rgb, sigma = model(inputs)

    rgb = rgb.reshape(list(pts.shape[:-1]) + [3])
    sigma = sigma.reshape(list(pts.shape[:-1]))
    # print(f"rgb: {rgb.shape}, sigma: {sigma.shape}")
    # logging.info(f"rgb {rgb.shape}\n sigma {sigma}")

    rgb_map, depth_map, acc_map, weights = volume_render(rays_d, rgb, sigma, pts, z_vals, white_bkgd=args.white_bkgd,
                                                         raw_noise_std=args.raw_noise_std, testing=not model.training)

    results = {
        "rgb_coarse": rgb_map,
        "depth_coarse": depth_map,
        "acc_coarse": acc_map,
    }

    if N_importance > 0:

        z_samples = sample_pdf((z_vals[..., 1:] + z_vals[..., :-1])*.5, weights[..., 1:-1], N_importance, model.training, not perturb, device=device)
        z_samples = z_samples.detach()
        z_vals = torch.concat([z_vals, z_samples], -1)
        z_vals, _ = torch.sort(z_vals, -1)
        
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]

        inputs = pts.reshape([-1, 3])
        inputs = posenc(args.L, inputs)

        if args.use_dirs:
            viewdirs = dirs[..., None, :].expand_as(pts).reshape([-1, 3])
            viewdirs = posenc(args.L_dirs, viewdirs)
            rgb, sigma = batchify(model_fine, inputs, viewdirs, use_dirs=True)
            # rgb, sigma = model_fine(inputs, viewdirs)
        else:
            rgb, sigma = batchify(model_fine, inputs, use_dirs=False)
            # rgb, sigma = model_fine(inputs)

        rgb = rgb.reshape(list(pts.shape[:-1]) + [3])
        sigma = sigma.reshape(list(pts.shape[:-1]))

        rgb_map, depth_map, acc_map, weights = volume_render(rays_d, rgb, sigma, pts, z_vals, white_bkgd=args.white_bkgd)
        # print(f"?????????? {rgb_map.dtype}, {sigma.dtype}")
        # logging.info(f"weights: {weights}")

        results.update({
            "rgb_fine": rgb_map,
            "depth_fine": depth_map,
            "acc_fine": acc_map,
        })

    return results


# class Args():
#     L = 10
#     L_dirs = 4
#     use_dirs = True

# args = Args
# from model.NeRF import NeRF
# model = NeRF(hidden_size=10)
# model_fine = NeRF(hidden_size=10)

# rays_o, rays_d = get_rays(5, 5, 1, torch.rand(4, 4))

# render_rays(args, model, rays_o, rays_d, 0., 1., 64, True, model_fine, 128)
