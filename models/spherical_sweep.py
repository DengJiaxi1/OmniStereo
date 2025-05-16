import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from typing import List
from ocamcamera import OcamCamera

from models.my_ocamcamera import myOcamCamera

class SphericalSweeping(object):
    def __init__(self, ocams:List[OcamCamera], poses:List[np.ndarray], h:int, w:int):
        """ SphericalSweeping module.

        Parameters
        ----------
        root_dir : str
            root directory includes poses.txt, ocam{i}.txt where i = [1,2,3,4]
        h : int
            output image height
        w : int
            output image width
        fov : float
            field of view of camera in degree
        """
        self.h = h
        self.w = w
        # load poses T cam <- world
        poses_cw = self.load_poses(poses)
        self._Tcw_tensor = [torch.tensor(T, dtype=torch.float32).cuda() for T in poses_cw]

        # rig center = [0., 0., 0.]

        # load ocam calibration data
        self._ocams = []
        for ocam in ocams:
            self._ocams.append(myOcamCamera(ocam))

        self.pts = torch.tensor(self.spherical_grid_3Dpoints(h, w), dtype=torch.float32).view(1, 1, h, w, 3).cuda()

    def load_poses(self, poses:List[np.ndarray]):
        """Calculate pose T cam <- world \in SE(3)"""
        Tcw = []
        for pose in poses:
            T = np.eye(4)  # T world <- cam
            R = Rot.from_rotvec(pose[:3]).as_matrix()
            T[:3, :3] = R
            T[:3, 3] = pose[3:]
            Tcw.append(np.linalg.inv(T))
        return Tcw

    def spherical_grid(self, h, w):
        p = 2 * np.pi / w
        th = np.pi / h
        phi = [-np.pi + (i + 0.5) * p for i in range(w)]
        theta = [-np.pi / 2 + (i + 0.5) * th for i in range(h)]
        phi_xy, theta_xy = np.meshgrid(phi, theta, sparse=False, indexing='xy')
        return phi_xy, theta_xy

    def spherical_grid_3Dpoints(self, h, w):
        phi_xy, theta_xy = self.spherical_grid(h, w)
        pts = np.stack([np.sin(phi_xy) * np.cos(theta_xy), np.sin(theta_xy), np.cos(phi_xy) * np.cos(theta_xy)], axis=2)
        return pts

    def get_grid(self, idx: int, depth):
        device = depth.device
        Tcw = self._Tcw_tensor[idx].to(device)

        # depth shape[B,N,H,W]
        b = depth.shape[0]
        n = depth.shape[1]

        # points from rig center [0., 0., 0.]
        pts =  (self.pts.to(device) * depth.unsqueeze(-1)).reshape(-1, 3)

        # add color from camera
        pts_c = Tcw[:3, :3].mm(pts.t()) + Tcw[:3, 3].unsqueeze(1)
        mapx, mapy = self._ocams[idx].world2cam(pts_c)

        # for grid_sample
        mapx = mapx.reshape(b, n, self.h, self.w)
        mapy = mapy.reshape(b, n, self.h, self.w)
        mapx = 2 * mapx / self._ocams[idx].width - 1
        mapy = 2 * mapy / self._ocams[idx].height - 1
        grid = torch.stack([mapx, mapy, torch.zeros_like(mapy)], dim=-1)

        return grid

