import numpy as np
import cv2

def write_depth_png(path, depth, gt_min=None, gt_max=None):
    if gt_min is None or gt_max is None:
        depth = np.log(depth - np.min(depth) + 1)
        depth = 255 * depth / np.max(depth)
    else:
        depth = np.clip(depth, gt_min, gt_max)
        depth = np.log(depth - gt_min + 1)
        depth = 255 * depth / np.log(gt_max - gt_min + 1)
    depth = np.clip(depth, 0, 255)
    depth = depth.astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    cv2.imwrite(path, depth)
