import torch
from tap import Tap
import json
from torch.utils import data
import cv2
from os.path import join
import time
import numpy as np

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def get_render_poses():
    return torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

class Dataset(data.Dataset):
    def __init__(self, args: Tap, split: str) -> None:
        super().__init__()

        base_dir = args.base_data

        with open(join(base_dir, f"transforms_{split}.json"), 'r') as f:
            data_json = json.load(f)
        
        self.data_json = data_json
        
        camera_angle_x = data_json["camera_angle_x"]
        
        print(f"Loading {split} dataset from {join(base_dir, f'transforms_{split}')}...")
        t = time.time()
        self.imgs = []
        self.poses = []
        for frame in data_json["frames"]:
            img = cv2.imread(join(base_dir, frame["file_path"][2:]+".png"), cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            img = (img / 255.).astype(np.float32)
            # if args.white_bkgd:
            #     img = img[..., :3]*img[..., -1:] + (1.-img[..., -1:])
            # else:
            #     img = img[..., :3]
            pose = np.array(frame["transform_matrix"])
            self.imgs.append(img)
            self.poses.append(pose)
        
        self.H, self.W = self.imgs[0].shape[:2]
        self.focal = .5 * self.W / np.tan(.5 * camera_angle_x)

        if args.half_res:

            for i in range(len(self.imgs)):
                self.imgs[i] = cv2.resize(self.imgs[i], (400, 400), interpolation=cv2.INTER_AREA)

            self.H = self.H // 2
            self.W = self.W // 2
            self.focal = self.focal / 2
        
        for i in range(len(self.imgs)):
            if args.white_bkgd:
                self.imgs[i] = self.imgs[i][..., :3]*self.imgs[i][..., -1:] + (1. - self.imgs[i][..., -1:])
            else:
                self.imgs[i] = self.imgs[..., :3]
        
        print(f"Loaded. Elapsed time: {time.time() - t:.3f}s")
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img = self.imgs[index]
        pose = self.poses[index]

        return img, pose, self.H, self.W, self.focal, \
              self.data_json["frames"][index]["file_path"].split('/')[-1] + ".png"

# import os
# os.chdir("/home/x/xie77777/codes/mynerf")
# import sys
# print(sys.path)
# sys.path.append("/home/x/xie77777/codes/mynerf")
# from config.config_lego import Args
# from utils import visualize
# from torch.utils.data.dataloader import DataLoader
# from torchvision.transforms.functional import center_crop
# dataset = Dataset(Args, "train")
# dataloader = DataLoader(dataset, 1)
# for img, pose, H, W, focal, filename in dataloader:
#     print(img.shape, pose.shape, H.shape, W.shape, focal.shape)
#     img = img.squeeze(0).permute([2, 0, 1])
#     H = H.squeeze(0).numpy()
#     W = W.squeeze(0).numpy()
#     img = center_crop(img, [H // 2, W // 2])
#     img = img.permute([1, 2, 0])
#     visualize(1, img, "./log", filename[0])
