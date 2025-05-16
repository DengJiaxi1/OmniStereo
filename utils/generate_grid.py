import torch
import numpy as np
from ocamcamera import OcamCamera

def gen_ocam2cassini_grid(ocam:OcamCamera, ca_h:int, ca_w:int, rotate_matrix:np.ndarray):

    iH = ocam._img_size[0]
    iW = ocam._img_size[1]

    p = np.pi / ca_w
    th = 2 * np.pi / ca_h
    phi = [-np.pi / 2 + (i + 0.5) * p for i in range(ca_w)]
    theta = [-np.pi + (i + 0.5) * th for i in range(ca_h)]
    phi_ca_map, theta_ca_map = np.meshgrid(phi, theta, sparse=False, indexing='xy')

    x = np.sin(phi_ca_map)
    y = np.cos(phi_ca_map) * np.sin(theta_ca_map)
    z = np.cos(phi_ca_map) * np.cos(theta_ca_map)
    point3D = np.stack([x, y, z]).reshape(3, -1)

    point3D = rotate_matrix.dot(point3D)
    mapx, mapy = ocam.world2cam(point3D)
    mapx = mapx.reshape(ca_h, ca_w)
    mapy = mapy.reshape(ca_h, ca_w)
    mapx = 2* mapx / iW - 1
    mapy = 2 * mapy / iH - 1
    grid = torch.from_numpy(np.stack([mapx, mapy], axis=-1)).unsqueeze(0)

    return grid


def gen_ocam2erp_grid(ocam:OcamCamera, h:int, w:int, rotate_matrix:np.ndarray):

    iH = ocam._img_size[0]
    iW = ocam._img_size[1]

    p = 2 * np.pi / w
    th = np.pi / h
    phi = [-np.pi + (i + 0.5) * p for i in range(w)]
    theta = [-np.pi / 2 + (i + 0.5) * th for i in range(h)]
    phi_xy, theta_xy = np.meshgrid(phi, theta, sparse=False, indexing='xy')

    point3D = np.stack([np.sin(phi_xy) * np.cos(theta_xy), np.sin(theta_xy),
                        np.cos(phi_xy) * np.cos(theta_xy)]).reshape(3, -1)

    point3D = rotate_matrix.dot(point3D)
    mapx, mapy = ocam.world2cam(point3D)
    mapx = mapx.reshape(h, w)
    mapy = mapy.reshape(h, w)
    mapx = 2 * mapx / iW - 1
    mapy = 2 * mapy / iH - 1
    grid = torch.from_numpy(np.stack([mapx, mapy], axis=-1)).unsqueeze(0)

    return grid

def gen_cassini2erp_grid(erp_h:int, erp_w:int):
    theta_erp_start = np.pi - (np.pi / erp_w)
    theta_erp_end = -np.pi
    theta_erp_step = 2 * np.pi / erp_w
    theta_erp_range = np.arange(theta_erp_start, theta_erp_end, -theta_erp_step)

    phi_erp_start = 0.5 * np.pi - (0.5 * np.pi / erp_h)
    phi_erp_end = -0.5 * np.pi
    phi_erp_step = np.pi / erp_h
    phi_erp_range = np.arange(phi_erp_start, phi_erp_end, -phi_erp_step)

    theta_erp_map, phi_erp_map = np.meshgrid(theta_erp_range, phi_erp_range, indexing='xy')

    theta_cassini_map = np.arctan2(np.tan(phi_erp_map), np.cos(theta_erp_map))
    phi_cassini_map = np.arcsin(np.cos(phi_erp_map) * np.sin(theta_erp_map))

    grid_x = torch.Tensor(
        np.clip(-phi_cassini_map / (0.5 * np.pi), -1, 1)).unsqueeze(-1)
    grid_y = torch.Tensor(
        np.clip(-theta_cassini_map / np.pi, -1, 1)).unsqueeze(-1)
    grid = torch.cat([grid_x, grid_y], dim=-1).unsqueeze(0)

    return grid


def crop_grid(grid:torch.Tensor, cassini_h:int, cassini_w:int, crop_top:int, crop_bottom:int, crop_left:int, crop_right:int):
    grid_x = grid[..., 0:1]
    grid_y = grid[..., 1:2]
    grid_x = (grid_x + 1) * (cassini_w) / 2
    grid_y = (grid_y + 1) * (cassini_h) / 2
    grid_x = (grid_x - crop_left) * 2 / (crop_right - crop_left) - 1
    grid_y = (grid_y - crop_top) * 2 / (crop_bottom - crop_top) - 1
    grid_x = torch.clamp(grid_x, min=-1, max=1)
    grid_y = torch.clamp(grid_y, min=-1, max=1)
    return torch.cat([grid_x, grid_y], dim=-1)


def gen_cassisi_phi(cassini_h, cassini_w):
    phi_step = torch.pi / cassini_w
    phi_start = -0.5 * torch.pi + phi_step / 2
    phi_end = 0.5 * torch.pi
    phi_range = torch.arange(phi_start, phi_end, phi_step)
    phi_map = phi_range.view(1, -1).repeat(cassini_h, 1)
    return phi_map


def gen_erp_unit_sphere(erp_h, erp_w):
    phi_erp_step = 2 * torch.pi / erp_w
    phi_erp_start = -torch.pi + phi_erp_step / 2
    phi_erp_end = torch.pi
    phi_erp_range = torch.arange(phi_erp_start, phi_erp_end, phi_erp_step)
    theta_erp_step = torch.pi / erp_h
    theta_erp_start = -torch.pi / 2 + theta_erp_step / 2
    theta_erp_end = torch.pi / 2
    theta_erp_range = torch.arange(theta_erp_start, theta_erp_end, theta_erp_step)
    phi_erp_map, theta_erp_map = torch.meshgrid(phi_erp_range, theta_erp_range, indexing='xy')
    x_sphere_erp = torch.sin(phi_erp_map) * torch.cos(theta_erp_map)
    y_sphere_erp = torch.sin(theta_erp_map)
    z_sphere_erp = torch.cos(phi_erp_map) * torch.cos(theta_erp_map)
    sphere_erp = torch.stack((x_sphere_erp, y_sphere_erp, z_sphere_erp), dim=-1)
    return sphere_erp

