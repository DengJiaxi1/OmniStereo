import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from typing import List
from ocamcamera import OcamCamera

from utils.calibration import calibration
from utils.generate_grid import gen_ocam2cassini_grid, gen_ocam2erp_grid, gen_cassini2erp_grid, crop_grid, gen_cassisi_phi, gen_erp_unit_sphere
from utils.iter_pixels import iter_pixels
from models.stereo_matching import StereoMatching
from models.fusion import Fusion
from models.refine import Refine


class OmniStereo(nn.Module):
    def __init__(self, ocams: List[OcamCamera], poses: List[np.ndarray], max_disp: int, max_depth: float,
                 erp_h: int = 320, erp_w: int = 640, att_only: bool = False, only_disp: bool = False):
        super(OmniStereo, self).__init__()
        assert len(ocams) == len(poses)

        self.model_disparity = StereoMatching(maxdisp=max_disp, att_weights_only=att_only)
        self.model_fusion = Fusion(channels=32)
        self.model_refine = Refine(ocams, poses, h=erp_h, w=erp_w)

        self.cam_num = len(ocams)
        self.max_depth = max_depth
        self.erp_h = erp_h  # cassini_w
        self.erp_w = erp_w  # cassini_h
        self.only_disp = only_disp
        self.ocam_h = ocams[0].height
        self.ocam_w = ocams[0].width

        # generate grids and masks
        baseline = []
        grids_ocam2cassini_l = []
        grids_ocam2cassini_r = []
        masks_cassini = []
        masks_erp = []

        # calibration
        rotate_left, rotate_right = calibration(poses)

        # generate grids and masks to construct cassini stereo pairs
        for i in range(self.cam_num):
            baseline.append(np.linalg.norm(poses[i][3:] - poses[(i + 1) % self.cam_num][3:]))
            grid_ocam2cassini_l = gen_ocam2cassini_grid(ocams[i], self.erp_w, self.erp_h, rotate_left[i])
            grid_ocam2cassini_r = gen_ocam2cassini_grid(ocams[(i + 1) % self.cam_num], self.erp_w, self.erp_h, rotate_right[i])
            grid_ocam2erp_l = gen_ocam2erp_grid(ocams[i], self.erp_h, self.erp_w, rotate_left[i])
            grid_ocam2erp_r = gen_ocam2erp_grid(ocams[(i + 1) % self.cam_num], self.erp_h, self.erp_w, rotate_right[i])
            grids_ocam2cassini_l.append(grid_ocam2cassini_l)
            grids_ocam2cassini_r.append(grid_ocam2cassini_r)

            valid_l = transforms.ToTensor()(ocams[i].valid_area())
            valid_r = transforms.ToTensor()(ocams[(i + 1) % self.cam_num].valid_area())
            valid_cassini_l = F.grid_sample(valid_l.unsqueeze(0), grid_ocam2cassini_l, align_corners=False)
            valid_cassini_r = F.grid_sample(valid_r.unsqueeze(0), grid_ocam2cassini_r, align_corners=False)
            valid_erp_l = F.grid_sample(valid_l.unsqueeze(0), grid_ocam2erp_l, align_corners=False)
            valid_erp_r = F.grid_sample(valid_r.unsqueeze(0), grid_ocam2erp_r, align_corners=False)

            mask_cassini = valid_cassini_l * valid_cassini_r
            masks_cassini.append((mask_cassini < 0.5).repeat(1, 3, 1, 1))
            mask_erp = valid_erp_l * valid_erp_r
            masks_erp.append((mask_erp < 0.5))

        self.baseline = torch.tensor(baseline, dtype=torch.float32).view(4, 1, 1, 1).cuda()
        grids_ocam2cassini_l_c = torch.cat(grids_ocam2cassini_l, dim=0)
        grids_ocam2cassini_r_c = torch.cat(grids_ocam2cassini_r, dim=0)
        masks_cassini_c = torch.stack(masks_cassini, dim=0)  # [4,1,3,cah,caw]
        self.masks_erp_c = torch.cat(masks_erp, dim=0).cuda()  # [4,1,eh,ew]

        # crop
        mask_index = torch.nonzero(~masks_cassini_c)
        self.crop_top = torch.min(mask_index[:, -2]).item()
        self.crop_bottom = torch.max(mask_index[:, -2]).item()
        self.crop_left = torch.min(mask_index[:, -1]).item()
        self.crop_right = torch.max(mask_index[:, -1]).item()
        if (self.crop_bottom - self.crop_top) % 32 > 0:
            self.crop_top -= ((32 - (self.crop_bottom - self.crop_top) % 32) // 2)
            self.crop_bottom += (32 - (self.crop_bottom - self.crop_top) % 32)
        if (self.crop_right - self.crop_left) % 32 > 0:
            self.crop_left -= ((32 - (self.crop_right - self.crop_left) % 32) // 2)
            self.crop_right += (32 - (self.crop_right - self.crop_left) % 32)
        self.crop_h = self.crop_bottom - self.crop_top
        self.crop_w = self.crop_right - self.crop_left

        self.grids_ocam2cassini_l_c = grids_ocam2cassini_l_c[:, self.crop_top:self.crop_bottom, self.crop_left:self.crop_right].cuda()  # [4,ch,cw,2]
        self.grids_ocam2cassini_r_c = grids_ocam2cassini_r_c[:, self.crop_top:self.crop_bottom, self.crop_left:self.crop_right].cuda()  # [4,ch,cw,2]
        self.masks_cassini_c = masks_cassini_c[..., self.crop_top:self.crop_bottom, self.crop_left:self.crop_right].cuda()  # [4,1,3,ch,cw]
        self.grid_cassini2erp = crop_grid(gen_cassini2erp_grid(self.erp_h, self.erp_w), self.erp_w, self.erp_h, self.crop_top, self.crop_bottom, self.crop_left, self.crop_right).cuda()  # [1,eh,ew,2]

        # Phi for Cassini
        phi_l_map = gen_cassisi_phi(cassini_h=erp_w, cassini_w=erp_h)
        self.phi_l_map = phi_l_map[self.crop_top:self.crop_bottom, self.crop_left:self.crop_right].cuda()  # [ch,cw]

        # unit sphere for ERP
        self.sphere_erp = gen_erp_unit_sphere(erp_h, erp_w).cuda()  # [eh,ew,3]

        # rotate
        rotate_list = []
        for i in range(self.cam_num):
            r = Rot.from_rotvec(poses[i][:3]).as_matrix() @ rotate_left[i]
            rotate_list.append(torch.tensor(r.T, dtype=torch.float32))
        self.rotate = torch.stack(rotate_list, dim=0).cuda() # [4,3,3]

        # trans
        trans_list = []
        for pose in poses:
            t = torch.tensor(pose[3:], dtype=torch.float32).view(1, 3)
            trans_list.append(t)
        self.trans = torch.stack(trans_list, dim=0).cuda()  # [4,1,3]


    def forward(self, imgs: list):
        assert len(imgs) == self.cam_num
        batch_size = imgs[0].shape[0]
        device = imgs[0].device

        # fisheye -> Cassini
        img_c = torch.stack(imgs, dim=0).view(4, -1, self.ocam_h, self.ocam_w)  # [4,b,3,oh,ow] -> [4,b*3,oh,ow]
        left_c = F.grid_sample(img_c, self.grids_ocam2cassini_l_c.to(device),
                               align_corners=False).view(4, batch_size, -1, self.crop_h, self.crop_w)
        # [4,b*3,ch,cw] -> [4,b,3,ch,cw]
        right_c = F.grid_sample(torch.roll(img_c, -1, dims=0), self.grids_ocam2cassini_r_c.to(device),
                                align_corners=False).view(4, batch_size, -1, self.crop_h, self.crop_w)
        # [4,b*3,ch,cw] -> [4,b,3,ch,cw]

        # outside of fov
        masks_cassini_c = self.masks_cassini_c.to(device).repeat(1, batch_size, 1, 1, 1)  # [4,1,3,ch,cw] -> [4,b,3,ch,cw]
        left_c[masks_cassini_c] = 0
        right_c[masks_cassini_c] = 0

        # stereo matching
        # input:[4,b,3,ch,cw] -> [4*b,3,ch,cw]; output:[4*b,ch,cw], [4*b,ch,cw], [4*b,1,ch,cw]
        pred_disp_c, pred_att_c, pred_conf_c = self.model_disparity(left_c.view(4 * batch_size, -1, self.crop_h, self.crop_w), right_c.view(4 * batch_size, -1, self.crop_h, self.crop_w))

        # disp -> depth
        pred_disp_c = torch.clamp(pred_disp_c, min=torch.finfo(pred_disp_c.dtype).eps).view(4, batch_size, self.crop_h, self.crop_w)
        # [4*b,ch,cw] -> [4,b,ch,cw]
        d = pred_disp_c * torch.pi / self.erp_h
        depth_l_c = self.baseline.to(device) * torch.sin(self.phi_l_map.to(device) - d + torch.pi / 2) / torch.sin(d)
        depth_l_c = torch.clamp(depth_l_c, min=0, max=self.max_depth)

        # Cassini -> ERP
        grid_cassini2erp = self.grid_cassini2erp.to(device)  # [1,eh,ew,2]
        depth_l_c_erp = F.grid_sample(depth_l_c.view(1, -1, self.crop_h, self.crop_w), grid_cassini2erp, mode='bilinear',
                                      align_corners=False, padding_mode='border').view(4, batch_size, self.erp_h, self.erp_w)
        # [4,b,ch,cw] -> [1,4*b,ch,cw] -> [1,4*b,eh,ew] -> [4,b,eh,ew]
        conf_c_erp = F.grid_sample(pred_conf_c.view(1, -1, self.crop_h, self.crop_w), grid_cassini2erp, mode='bilinear',
                                   align_corners=False, padding_mode='border').view(4, batch_size, self.erp_h, self.erp_w)
        # [4*b,1,ch,cw] -> [1,4*b,ch,cw] -> [1,4*b,eh,ew] -> [4,b,eh,ew]

        # outside of fov
        masks_erp_c = self.masks_erp_c.to(device).repeat(1, batch_size, 1, 1)  # [4,1,eh,ew] -> [4,b,eh,ew]
        depth_l_c_erp[masks_erp_c] = 0
        conf_c_erp[masks_erp_c] = 0

        # align to center
        points_0 = self.sphere_erp.to(device) * depth_l_c_erp.unsqueeze(-1)  # sphere:[eh,ew,3]; depth:[4,b,eh,ew]->[4,b,eh,ew,1]; points_0:[4,b,eh,ew,3]
        points_1 = torch.bmm(points_0.view(4, -1, 3), self.rotate.to(device)) + self.trans.to(device)  # rotate:[4,3,3]; trans:[4,1,3]
        points_1 = points_1.view(4 * batch_size, self.erp_h, self.erp_w, 3)  # [4,b*eh*ew,3] -> [4*b,eh,ew,3]
        depth_1 = torch.linalg.norm(points_1, dim=-1)

        phi_1_map = torch.arctan2(points_1[:, :, :, 0], points_1[:, :, :, 2])
        theta_1_map = torch.arcsin(torch.clamp(points_1[:, :, :, 1] / depth_1, -1, 1))

        i_1 = torch.clamp(torch.floor(self.erp_h * (theta_1_map / torch.pi + 0.5)), 0, self.erp_h - 1)
        j_1 = torch.clamp(torch.floor(self.erp_w * (phi_1_map / (2 * torch.pi) + 0.5)), 0, self.erp_w - 1)

        depth_trans, conf_trans = iter_pixels.apply(depth_l_c_erp.view(4 * batch_size, self.erp_h, self.erp_w), depth_1, conf_c_erp.view(4 * batch_size, self.erp_h, self.erp_w), i_1, j_1)
        depth_trans[depth_trans == 100000] = 0
        depth_trans = torch.clamp(depth_trans, 0, self.max_depth)

        depth_trans = depth_trans.view(4, batch_size, self.erp_h, self.erp_w).transpose(0, 1)
        # [4*b,eh,ew] -> [4,b,eh,ew] -> [b,4,eh,ew]
        conf_trans = conf_trans.view(4, batch_size, self.erp_h, self.erp_w).transpose(0, 1)
        # [4*b,eh,ew] -> [4,b,eh,ew] -> [b,4,eh,ew]

        # only estimate disparity
        if self.only_disp:
            depth_concat = torch.cat((depth_trans[:, 2, :, 0:160], depth_trans[:, 3, :, 160:320], depth_trans[:, 0, :, 320:480], depth_trans[:, 1, :, 480:640]), dim=-1)
            if self.training:
                return depth_concat, None, pred_disp_c.transpose(0,1), pred_att_c.view(4, batch_size, self.crop_h, self.crop_w).transpose(0,1)
            else:
                return depth_concat, None, depth_trans, conf_trans

        # fusion
        pred_fusion = self.model_fusion(depth_trans, conf_trans)  # [b,1,eh,ew]
        pred_fusion = torch.clamp(pred_fusion, 0, self.max_depth)

        # refine
        pred_refine = self.model_refine(imgs, pred_fusion)  # [b,eh,ew]
        pred_refine = torch.clamp(pred_refine, 0, self.max_depth)

        if self.training:
            return pred_refine, pred_fusion.squeeze(1), pred_disp_c.transpose(0,1), pred_att_c.view(4, batch_size, self.crop_h, self.crop_w).transpose(0,1)
        else:
            return pred_refine, pred_fusion.squeeze(1), depth_trans, conf_trans
