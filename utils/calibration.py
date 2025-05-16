import numpy as np
from scipy.spatial.transform import Rotation as Rot


def calibration(poses):
    rotate_left = []
    rotate_right = []

    for i in range(len(poses)):
        R1 = Rot.from_rotvec(poses[i][:3]).as_matrix()
        T1 = poses[i][3:]
        R2 = Rot.from_rotvec(poses[(i + 1) % len(poses)][:3]).as_matrix()
        T2 = poses[(i + 1) % len(poses)][3:]

        R = R1 @ R2.T
        T = R2.T @ (T1 - T2)
        om = Rot.from_matrix(R).as_rotvec()

        r_l = Rot.from_rotvec(om / 2).as_matrix()
        r_r = Rot.from_rotvec(-om / 2).as_matrix()
        t = r_r @ T

        if np.abs(t[0]) > np.abs(t[1]):
            uu = np.array([1, 0, 0])
        else:
            uu = np.array([0, 1, 0])
        if np.dot(uu, t) < 0:
            uu = -uu

        ww = np.cross(t, uu)
        ww = ww / np.linalg.norm(ww)
        ww = np.arccos(np.abs(np.dot(t, uu)) / (np.linalg.norm(t) * np.linalg.norm(uu))) * ww
        R_rec = Rot.from_rotvec(ww).as_matrix()

        R_L = R_rec @ r_l
        R_R = R_rec @ r_r
        rotate_left.append(R_L.T)
        rotate_right.append(R_R.T)

    return rotate_left, rotate_right


if __name__ == '__main__':
    poses = [np.array([0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 60.000000]),
             np.array([0.000000, 1.570796326794897, 0.000000, 60.000000, 0.000000, 0.000000]),
             np.array([0.000000, 3.141592653589793, 0.000000, 0.000000, 0.000000, -60.000000]),
             np.array([0.000000, -1.570796326794897, 0.000000, -60.000000, 0.000000, 0.000000])]
    rotate_left, rotate_right = calibration(poses)
    for i in range(len(rotate_left)):
        print(rotate_left[i], rotate_right[i])
