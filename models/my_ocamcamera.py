import torch
import math
from ocamcamera import OcamCamera


class myOcamCamera:

    def __init__(self, ocam: OcamCamera):

        # polynomial coefficients for the inverse mapping function
        self._invpol = ocam._invpol
        # center: "row" and "column", starting from 0 (C convention)
        self._xc = ocam._xc
        self._yc = ocam._yc
        # _affine parameters "c", "d", "e"
        self._affine = ocam._affine
        # field of view
        self._fov = torch.tensor(ocam._fov)
        self.height:int = ocam.height
        self.width:int  = ocam.width
        self.thresh_theta = torch.deg2rad(self._fov / 2) - math.pi / 2

    def world2cam(self, point3D):
        # input shape: [3, N]
        assert point3D.shape[0] == 3

        norm = torch.sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1])

        theta = -torch.atan2(point3D[2], norm)
        invnorm = 1 / norm

        rho = torch.full_like(theta, self._invpol[0])
        tmp_theta = theta.clone()
        for i in range(1, len(self._invpol)):
            rho += self._invpol[i] * tmp_theta
            tmp_theta *= theta

        u = point3D[0] * invnorm * rho
        v = point3D[1] * invnorm * rho
        point2D_valid_0 = v * self._affine[2] + u + self._yc
        point2D_valid_1 = v * self._affine[0] + u * self._affine[1] + self._xc

        outside_flag = theta > self.thresh_theta
        point2D_valid_0[outside_flag] = -1
        point2D_valid_1[outside_flag] = -1

        return point2D_valid_0, point2D_valid_1
