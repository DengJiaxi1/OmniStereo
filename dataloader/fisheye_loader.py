import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
from typing import List

__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


def color_normalize(normalize=__imagenet_stats):
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    return transforms.Compose(t_list)


def image_loader(path):
    return Image.open(path).convert('RGB')


def disp_loader(path):
    return np.load(path)['arr_0'].astype(np.float32)


def tiff_loader(path):
    depth = Image.open(path)
    depth.seek(1)
    depth = np.array(depth).astype(np.float32)
    depth = np.reciprocal(depth)
    return depth


def exr_loader(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    depth = depth[:, :, 0]
    return depth


class fisheyeDataset(Dataset):
    def __init__(self, data_path: str, filename_txt: str, cam_list: List[str],
                 depth_folder: str = None, disp_folder: str = None, max_depth: float = 1000., height: int = 320):

        # Initialization
        super(fisheyeDataset, self).__init__()

        self.data_path = data_path

        self.datasets = []
        self.imgs = []
        self.depths = []
        self.disps = []
        self.dataset_list = []

        with open(filename_txt) as f:
            data = f.read()
        filenames = data.strip().split('\n')
        for filename in filenames:
            filename_ = filename.split(' ')
            self.datasets.append(filename_[0])
            if filename_[0] not in self.dataset_list:
                self.dataset_list.append(filename_[0])
            self.imgs.append(filename_[1])
            if depth_folder is not None:
                self.depths.append(filename_[2])
            if disp_folder is not None:
                self.disps.append(filename_[3])

        if depth_folder is not None:
            ext = os.path.splitext(self.depths[0])[1]
            if ext == '.tiff':
                self.depth_loader = tiff_loader
            elif ext == '.exr':
                self.depth_loader = exr_loader

        self.cam_list = cam_list
        self.depth_folder = depth_folder
        self.disp_folder = disp_folder
        self.max_depth = max_depth
        self.height = height
        self.processed = color_normalize()  # transform of rgb images

    def __getitem__(self, index):
        imgs = []
        disps = []
        depth = []

        for cam in self.cam_list:
            img = image_loader(os.path.join(self.data_path, self.datasets[index], cam, self.imgs[index]))
            img = np.array(img)
            img = self.processed(img)
            imgs.append(img)

        if self.depth_folder is not None:
            depth = self.depth_loader(os.path.join(self.data_path, self.datasets[index], self.depth_folder, self.depths[index]))
            depth = np.clip(depth, 0, self.max_depth)
            pad = (self.height - depth.shape[0]) // 2
            if pad > 0:
                depth = np.pad(depth, ((pad, pad), (0, 0)), 'constant', constant_values=self.max_depth + 1)

        if self.disp_folder is not None:
            for cam in self.cam_list:
                disp = disp_loader(
                    os.path.join(self.data_path, self.datasets[index], self.disp_folder, cam, self.disps[index]))
                disps.append(disp)

        return {'imgs': imgs, 'depth': depth, 'disps': disps, 'dataset': self.datasets[index], 'name': self.imgs[index]}

    def __len__(self):
        return len(self.imgs)
