import os
import cv2
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
# device = "cpu"
from config.config_lego import Args
from utils.render import render_rays, get_rays
from utils.load_blender import Dataset, get_render_poses
from utils.utils import load_ckpt, save_ckpt, psnr_np, visualize, mkdir
from model.loss import criterion, psnr
from model.NeRF import NeRF
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import logging
from torchvision.transforms.functional import center_crop
import random

def train(args, epoch, img, pose, H, W, focal, model, optimizer, model_fine=None):
    with torch.set_grad_enabled(True):
        model.train()
        if model_fine is not None:
            model_fine.train()
        rays_o, rays_d = get_rays(H, W, focal, pose)

        if epoch <= args.precrop:
            H = H.numpy() // 2
            W = W.numpy() // 2
            img = torch.permute(img, [2, 0, 1])
            rays_o = torch.permute(rays_o, [2, 0, 1])
            rays_d = torch.permute(rays_d, [2, 0, 1])
            img = center_crop(img, [H, W])
            rays_o = center_crop(rays_o, [H, W])
            rays_d = center_crop(rays_d, [H, W])
            img = torch.permute(img, [1, 2, 0])
            rays_o = torch.permute(rays_o, [1, 2, 0])
            rays_d = torch.permute(rays_d, [1, 2, 0])

        shape = img.shape
        img = img.reshape([-1, 3]).to(device, dtype=dtype)
        rays_o = rays_o.reshape([-1, 3]).to(device, dtype=dtype)
        rays_d = rays_d.reshape([-1, 3]).to(device, dtype=dtype)

        perm = torch.randperm(rays_o.shape[0])
        idx = perm[0:args.N_rand]
        img = img[idx, :]
        rays_o = rays_o[idx, :]
        rays_d = rays_d[idx, :]

        # idx = np.random.randint(0, rays_o.shape[0])
        # img = img[idx:idx+args.N_rand, :]
        # rays_o = rays_o[idx:idx+args.N_rand, :]
        # rays_d = rays_d[idx:idx+args.N_rand, :]

        results = render_rays(args, model, rays_o, rays_d, 2., 6., args.N_samples, True, model_fine, args.N_importance)
        if args.N_importance > 0:
            loss = criterion(results["rgb_coarse"], img) + criterion(results["rgb_fine"], img)
            train_psnr = psnr(results["rgb_fine"], img)
        else:
            loss = criterion(results["rgb_coarse"], img)
            train_psnr = psnr(results["rgb_coarse"], img)
        
        if torch.isinf(train_psnr) or torch.isnan(train_psnr):
            logging.info(f"numerical error at {epoch}, loss {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if DEBUG:
        #     for key in results:
        #         if torch.isnan(results[key]).any() or torch.isinf(results[key]).any():
        #             logging.info(f"! [Numerical Error] at epoch {epoch}, {key} contains nan or inf.")

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (epoch / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
    
        loss = loss.detach().cpu().numpy()
        train_psnr = train_psnr.detach().cpu().numpy()
        return loss, train_psnr

        # for i in range(0, rays_o.shape[0], chunk):
        #     batched_rays_o = rays_o[i:i+chunk]
        #     batched_rays_d = rays_d[i:i+chunk]

        #     # batched_rays_d = batched_rays_d.to(device)
        #     # batched_rays_o = batched_rays_o.to(device)
        #     target = img[i:i+chunk].to(device, dtype=torch.float32)
        #     results = render_rays(args, model, batched_rays_o, batched_rays_d, 2., 6., args.N_samples, True, model_fine, args.N_importance)
        #     if args.N_importance > 0:
        #         loss = criterion(results["rgb_coarse"], target) + criterion(results["rgb_fine"], target)
        #     else:
        #         loss = criterion(results["rgb_coarse"], target)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
            # sum += loss.detach().cpu().numpy() * batched_rays_o.shape[0]

def render_out(
        args,
        dataloader,
        model,
        model_fine = None,
        size=None,
        vis: bool = False,
        logdir = None,
        epoch = None,
        mode: str = "test",
        writer: SummaryWriter=None,
):
    psnrlist = []
    dataiter = iter(dataloader)
    n_iter = len(dataloader)
    if size is not None:
        n_iter = size
    if epoch is None:
        epoch = "test"
    video_writer = None
    if mode == "test":
        video_writer = cv2.VideoWriter(logdir + "/render.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (400, 400))
    with torch.no_grad():
        for _ in enumerate(tqdm(range(0, n_iter), leave=False, ascii=True, position=1)):
            img, pose, H, W, focal, filename = next(dataiter)
            img = img.squeeze(0)
            pose = pose.squeeze(0)
            filename = filename[0]
            H = H.squeeze(0)
            W = W.squeeze(0)
            rays_o, rays_d = get_rays(H, W, focal, pose)
            shape = img.shape
            chunk = args.N_rand
            rays_o = rays_o.reshape([-1, 3]).to(device)
            rays_d = rays_d.reshape([-1, 3]).to(device)
            res = []
            for i in range(0, rays_o.shape[0], chunk):
                batched_rays_o = rays_o[i:i+chunk]
                batched_rays_d = rays_d[i:i+chunk]
                # batched_rays_d = batched_rays_d.to(device)
                # batched_rays_o = batched_rays_o.to(device)
                results = render_rays(args, model, batched_rays_o, batched_rays_d, 2., 6., args.N_samples, True, model_fine, args.N_importance)
                rgb = results["rgb_coarse"]
                if args.N_importance > 0:
                    rgb = results["rgb_fine"]
                res.append(rgb)

            res = torch.concat(res, 0)
            res = res.reshape(shape)
            if writer is not None:
                res = res.permute([2, 0, 1])
                writer.add_image("val", res, epoch)
                res = res.permute([1, 2, 0])
            res = res.detach().cpu().numpy()
            if vis:
                visualize(epoch, res, logdir, mode, filename, video_writer)
            psnrlist.append(psnr_np(res, img.numpy()))
        if video_writer is not None:
            video_writer.release()
    return np.array(psnrlist).mean()

    

def val(args, epoch, val_set, model, model_fine = None, vis: bool = False, logdir=None, writer: SummaryWriter = None):
    model.eval()
    if model_fine is not None:
        model_fine.eval()
    dataloader = DataLoader(val_set, 1, False, num_workers=2, pin_memory=True)
    return render_out(args, dataloader, model, model_fine, args.val_size, vis, logdir, epoch=epoch, mode="val", writer=writer)

def test(args, model, model_fine=None):
    model.eval()
    if model_fine is not None:
        model_fine.eval()
    test_set = Dataset(args, "val")
    dataloader = DataLoader(test_set, 1, True, num_workers=2, pin_memory=True)
    logdir = os.path.join(args.logpath, args.dataset, args.exp_name)
    render_poses = get_render_poses()
    video_writer = cv2.VideoWriter(logdir + "/paper_render.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (400, 400))
    with torch.no_grad():
        for c2w in tqdm(render_poses):
            rays_o, rays_d = get_rays(400, 400, test_set.focal, c2w)
            rays_o = rays_o.cuda()
            rays_d = rays_d.cuda()
            results = render_rays(args, model, rays_o, rays_d, 2., 6., args.N_samples, True, model_fine, args.N_importance)
            rgb = results["rgb_fine"]
            visualize(None, rgb.detach().cpu().numpy(), logdir, "test", "paper_render.mp4", video_writer)
    video_writer.release()
    return render_out(args, dataloader, model, model_fine, size=40, vis=True, logdir=logdir, mode="test")


def main():
    ########## __main__ ##########

    # init
    args = Args().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
        

    model = NeRF(L_embed_dir=args.L_dirs, L_embed_loc=args.L)
    model = model.to(device)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model_fine = None
    grad_vars = list(model.parameters())
    if args.N_importance > 0:
        model_fine = NeRF(L_embed_dir=args.L_dirs, L_embed_loc=args.L)
        model_fine = model_fine.to(device)
        # model_fine = nn.DataParallel(model_fine, device_ids=[0, 1])
        grad_vars += list(model_fine.parameters())

    lrate = args.lrate
    optimizer = optim.Adam(grad_vars, lr = args.lrate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)

    if args.test_only:
        if args.resume is None:
            print("No reference to the test model.")
            return
        checkpoint = load_ckpt(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        if args.N_importance > 0:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        print(f"psnr: {test(args, model, model_fine)}")
        return

    train_set, val_set = Dataset(args, "train"), Dataset(args, "val")
    # train_set = Dataset(args, "train")

    # load
    st_epoch = 0
    losslist = []
    psnrlist = []
    best_psnr = 0
    if args.resume is not None:
        print(f"loading checkpoint {args.resume}...")
        # st_epoch, best_psnr, model_state_dict, optimizer_state_dict, losslist, model_fine_state_dict = load_ckpt(args.resume_filename)
        checkpoint = load_ckpt(args.resume)
        st_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        losslist = checkpoint["losslist"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if args.N_importance > 0:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])

    print(f"starting from epoch {st_epoch+1}...")

    trainloader = DataLoader(train_set, 1, True, num_workers=2, pin_memory=True)
    train_iter = iter(trainloader)


    logdir = os.path.join(args.logpath, args.dataset, args.exp_name)
    mkdir(logdir)
    logging.basicConfig(filename=logdir+"/log.txt", level=logging.DEBUG)
    logger = logging.getLogger(logdir+"/log.txt")

    writer = SummaryWriter(log_dir=logdir, comment=args.dataset)

    logger.info(args)
    logger.info(model)
    if model_fine is not None:
        logger.info(model_fine)

    show_loss_epoch = len(trainloader)
    scheduler_step = args.nums_iters // 100

    for epoch in tqdm(range(st_epoch + 1, args.nums_iters + 1), ascii=True, position=0):
        if (epoch - 1) % len(trainloader) == 0:
            train_iter = iter(trainloader)
        img, pose, H, W, focal, _ = next(train_iter)
        img = img.squeeze(0)
        pose = pose.squeeze(0)
        H = H.squeeze(0)
        W = W.squeeze(0)
        focal = focal.squeeze(0)
        loss, train_psnr = train(args, epoch, img, pose, H, W, focal, model, optimizer, model_fine)

        # if epoch % scheduler_step == 0:
        #     scheduler.step()

        losslist.append(loss)
        psnrlist.append(train_psnr)
        # tqdm.write(f"loss: {loss:.6f}")
        # writer.add_scalar("loss", loss, epoch)
        # writer.add_scalar("train psnr", train_psnr, epoch)
        # logger.info(f"loss: {loss:.6f}")
        if epoch % args.ckpt_epoch == 0 and epoch > 0:
            save_ckpt(logdir+"/"+args.dataset+"_ckpt", args, epoch, best_psnr, model, optimizer, scheduler, losslist, model_fine)
        if epoch % show_loss_epoch == 0 and epoch > 0:
            logger.info(f"""avg loss: {sum(losslist[epoch-show_loss_epoch:epoch]) / show_loss_epoch : .6f}
                            avg train psnr: {sum(psnrlist[len(psnrlist)-show_loss_epoch:epoch]) / show_loss_epoch}""")
            writer.add_scalar("avg loss", sum(losslist[epoch-show_loss_epoch:epoch]) / show_loss_epoch, epoch)
            writer.add_scalar("avg train psnr", sum(psnrlist[len(psnrlist)-show_loss_epoch:epoch]) / show_loss_epoch, epoch)

        if epoch % args.val_epoch == 0 and epoch >= args.start_val:
            psnr = val(args, epoch, val_set, model, model_fine, vis=True, logdir=logdir, writer=writer)
            tqdm.write(f"avg psnr: {psnr:.6f}")
            logger.info(f"avg psnr: {psnr:.6f}")
            writer.add_scalar("avg psnr", psnr, epoch)
            if psnr > best_psnr:
                best_psnr = psnr
                save_ckpt(logdir+"/"+args.dataset+"_best", args, epoch, best_psnr, model, optimizer, scheduler, losslist, model_fine)
    print(f"best psnr {best_psnr} Done.")

if __name__ == "__main__":
    torch.set_default_dtype(dtype)
    main()