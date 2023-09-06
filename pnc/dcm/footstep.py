import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class Footstep(object):
    LEFT_SIDE = 0
    RIGHT_SIDE = 1

    def __init__(self):
        self._pos = np.zeros(3)
        self._quat = np.array([0., 0., 0., 1.])  # scalar-last
        self._rot = np.eye(3)
        self._iso = np.eye(4)
        self._side = -1

    @property
    def pos(self):
        return self._pos

    @property
    def quat(self):
        return self._quat

    @property
    def rot(self):
        return self._rot

    @property
    def iso(self):
        return self._iso

    @property
    def side(self):
        return self._side

    @pos.setter
    def pos(self, value):
        self._pos = np.copy(value)
        self._iso[0:3, 3] = np.copy(value)

    @quat.setter
    def quat(self, value):
        self._quat = np.copy(value)
        self._rot = R.from_quat(value).as_matrix()
        self._iso[0:3, 0:3] = np.copy(self._rot)

    @rot.setter
    def rot(self, value):
        self._quat = R.from_matrix(value).as_quat()
        self._rot = np.copy(value)
        self._iso[0:3, 0:3] = np.copy(self._rot)

    @iso.setter
    def iso(self, value):
        self._pos = np.copy(value[0:3, 3])
        self._quat = R.from_matrix(value[0:3, 0:3]).as_quat()
        self._rot = np.copy(value[0:3, 0:3])
        self._iso = np.copy(value)

    @side.setter
    def side(self, value):
        self._side = value


def interpolate(footstep1, footstep2, alpha=0.5):
    mid_foot = Footstep()
    mid_foot.pos = alpha * footstep1.pos + (1 - alpha) * footstep2.pos
    slerp = Slerp([0, 1], R.from_quat([footstep1.quat, footstep2.quat]))
    mid_foot.quat = slerp(alpha).as_quat()
    return mid_foot
