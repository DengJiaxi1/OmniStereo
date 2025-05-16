import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from ocamcamera import OcamCamera

from models.submodule import BasicConv, Conv2x, depth_regression
from models.spherical_sweep import SphericalSweeping


class FeatureExtraction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureExtraction, self).__init__()
        self.stem_2 = nn.Sequential(
                      BasicConv(in_channels, 24, kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(24, 24, 3, 1, 1, bias=False),
                      nn.BatchNorm2d(24), nn.ReLU())
        self.stem_4 = nn.Sequential(
                      BasicConv(24, 32, kernel_size=3, stride=2, padding=1),
                      nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                      nn.BatchNorm2d(32), nn.ReLU())
        self.spx_2 = Conv2x(32, 24, True)
        self.spx = nn.Sequential(nn.ConvTranspose2d(48, out_channels, kernel_size=4, stride=2, padding=1),)

    def forward(self, img):
        stem_2x = self.stem_2(img)
        stem_4x = self.stem_4(stem_2x)
        xspx = self.spx_2(stem_4x, stem_2x)
        spx_pred = self.spx(xspx)
        return spx_pred

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv2_up = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up_ = BasicConv(in_channels * 2, in_channels, deconv=True, is_3d=True, bn=True,
                                   relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv_final = nn.Conv3d(in_channels, 1, 3, 1, 1, bias=False)

        self.agg = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1), )

    def forward(self, x):
        conv1 = self.conv1(x)

        conv2 = self.conv2(conv1)

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg(conv1)

        conv = self.conv1_up_(conv1)
        conv = self.conv_final(conv)

        return conv


class Refine(nn.Module):
    def __init__(self, ocams:List[OcamCamera], poses:List[np.ndarray], h:int, w:int):
        super(Refine, self).__init__()
        self.h = h
        self.w = w
        self.c = 16
        self.n = 16
        self.feature_extraction = FeatureExtraction(3, self.c)
        self.sweep = SphericalSweeping(ocams, poses, h, w)
        self.fusion = nn.Sequential(
            nn.Conv3d(self.c * 4, self.c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.cost_regularization = hourglass(self.c)

    def forward(self, fisheye_rgbs, depth):
        dtype = fisheye_rgbs[0].dtype
        device = fisheye_rgbs[0].device
        batch_size = fisheye_rgbs[0].shape[0]

        depth_samples = depth * torch.arange(0.65, 1.45, 0.05, dtype=dtype, device=device).view(1, -1, 1, 1) # 16

        fisheye_rgbs_c = torch.cat(fisheye_rgbs, dim=0) # [4*b,3,oh,ow]
        feat_c = self.feature_extraction(fisheye_rgbs_c).unsqueeze(2) # [4*b,c,oh,ow] -> [4*b,c,1,oh,ow]

        # define empty cost volume
        costs = torch.zeros((batch_size, self.c * 4, self.n, self.h, self.w), dtype=dtype, device=device)

        # construct cost volume
        for i in range(4):
            with torch.no_grad():
                grids = self.sweep.get_grid(i, depth_samples)
            warps = F.grid_sample(feat_c[batch_size * i:batch_size * (i + 1)], grids, align_corners=False)
            costs[:, self.c * i : self.c * (i + 1), :, :, :] = warps

        # fusion
        costs = self.fusion(costs)

        # cost volume computation
        out = self.cost_regularization(costs)

        # regression
        pred = depth_regression(depth_samples, out)

        return pred
