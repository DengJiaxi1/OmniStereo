import os
import numpy as np
from typing import List
from ocamcamera import OcamCamera

def load_ocam(ocam_path: str, cam_list: List[str], fov: int):
    ocams = []
    poses = []
    for cam in cam_list:
        ocam_file = os.path.join(ocam_path, f'o{cam}.txt')
        ocam = OcamCamera(ocam_file, fov)
        ocams.append(ocam)
    with open(os.path.join(ocam_path, 'poses.txt')) as f:
        poses_data = f.readlines()
    for data in poses_data:
        pose = np.array(data.split(), dtype=float)
        pose[3:] /= 100  # cm -> m
        poses.append(pose)
    return ocams, poses