import numpy as np
from scipy.spatial.transform import Rotation as R

from util import liegroup


def x_rot(angle):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )


def iso_to_pose(iso_matrix):
    mat, pos = liegroup.TransToRp(iso_matrix)
    quat = rot_to_quat(mat)
    return pos, quat


def euler_to_rot(angles):
    # Euler ZYX to Rot
    # Note that towr has (x, y, z) order
    x = angles[0]
    y = angles[1]
    z = angles[2]
    ret = np.array(
        [
            np.cos(y) * np.cos(z),
            np.cos(z) * np.sin(x) * np.sin(y) - np.cos(x) * np.sin(z),
            np.sin(x) * np.sin(z) + np.cos(x) * np.cos(z) * np.sin(y),
            np.cos(y) * np.sin(z),
            np.cos(x) * np.cos(z) + np.sin(x) * np.sin(y) * np.sin(z),
            np.cos(x) * np.sin(y) * np.sin(z) - np.cos(z) * np.sin(x),
            -np.sin(y),
            np.cos(y) * np.sin(x),
            np.cos(x) * np.cos(y),
        ]
    ).reshape(3, 3)
    return np.copy(ret)


def euler_to_quat(angles):
    # Euler ZYX to Rot
    # Note that towr has (x, y, z) order
    return np.copy((R.from_euler("xyz", angles, degrees=False)).as_quat())


def quat_to_rot(quat):
    """
    Parameters
    ----------
    quat (np.array): scalar last quaternion

    Returns
    -------
    ret (np.array): SO3

    """
    return np.copy((R.from_quat(quat)).as_matrix())


def rot_to_quat(rot):
    """
    Parameters
    ----------
    rot (np.array): SO3

    Returns
    -------
    quat (np.array): scalar last quaternion

    """
    return np.copy(R.from_matrix(rot).as_quat())


def rot_to_euler(rot):
    """
    Parameters
    ----------
    rot (np.array): SO3

    Returns
    -------
    quat (np.array): scalar last quaternion

    """
    return np.copy(R.from_matrix(rot).as_euler("xyz"))


def quat_to_exp(quat):
    img_vec = np.array([quat[0], quat[1], quat[2]])
    w = quat[3]
    theta = 2.0 * np.arcsin(
        np.sqrt(
            img_vec[0] * img_vec[0] + img_vec[1] * img_vec[1] + img_vec[2] * img_vec[2]
        )
    )

    if np.abs(theta) < 1e-4:
        return np.zeros(3)
    ret = img_vec / np.sin(theta / 2.0)

    return np.copy(ret * theta)


def quat_to_euler(quat):
    return np.copy((R.from_quat(quat)).as_euler("xyz"))


def exp_to_quat(exp):
    theta = np.sqrt(exp[0] * exp[0] + exp[1] * exp[1] + exp[2] * exp[2])
    ret = np.zeros(4)
    if theta > 1e-4:
        ret[0] = np.sin(theta / 2.0) * exp[0] / theta
        ret[1] = np.sin(theta / 2.0) * exp[1] / theta
        ret[2] = np.sin(theta / 2.0) * exp[2] / theta
        ret[3] = np.cos(theta / 2.0)
    else:
        ret[0] = 0.5 * exp[0]
        ret[1] = 0.5 * exp[1]
        ret[2] = 0.5 * exp[2]
        ret[3] = 1.0
    return np.copy(ret)


def get_sinusoid_trajectory(start_time, mid_point, amp, freq, eval_time):
    dim = amp.shape[0]
    p, v, a = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    p = amp * np.sin(2 * np.pi * freq * (eval_time - start_time)) + mid_point
    v = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * (eval_time - start_time))
    a = (
        -amp
        * (2 * np.pi * freq) ** 2
        * np.sin(2 * np.pi * freq * (eval_time - start_time))
    )

    return p, v, a


def prevent_quat_jump(quat_des, quat_act):
    # print("quat_des:",quat_des)
    # print("quat_act:",quat_act)
    a = quat_des - quat_act
    b = quat_des + quat_act
    if np.linalg.norm(a) > np.linalg.norm(b):
        new_quat_act = -quat_act
    else:
        new_quat_act = quat_act

    return new_quat_act
