from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np

from util import liegroup


def linear_changing(ini, end, dur, curr_time):
    ret = ini + (end - ini) * 0.5 * (curr_time / dur)
    if curr_time > dur:
        ret = end

    return ret


def linear_changing_vel(ini, end, dur, curr_time):
    ret = (end - ini) / dur
    if curr_time > dur:
        ret = 0.

    return ret


def linear_changing_acc(ini, end, dur, curr_time):

    return 0.


def smooth_changing(ini, end, dur, curr_time):
    ret = ini + (end - ini) * 0.5 * (1 - np.cos(curr_time / dur * np.pi))
    if curr_time > dur:
        ret = end

    return ret


def smooth_changing_vel(ini, end, dur, curr_time):
    ret = (end - ini) * 0.5 * (np.pi / dur) * np.sin(curr_time / dur * np.pi)
    if curr_time > dur:
        ret = 0.

    return ret


def smooth_changing_acc(ini, end, dur, curr_time):
    ret = (end - ini) * 0.5 * (np.pi / dur) * (
        np.pi / dur) * np.cos(curr_time / dur * np.pi)
    if curr_time > dur:
        ret = 0.

    return ret


def iso_interpolate(T1, T2, alpha):
    p1 = T1[0:3, 3]
    R1 = T1[0:3, 0:3]
    p2 = T2[0:3, 3]
    R2 = T2[0:3, 0:3]

    slerp = Slerp([0, 1], R.from_matrix([R1, R2]))

    p_ret = alpha * (p1 + p2)
    R_ret = slerp(alpha).as_matrix()

    return liegroup.RpToTrans(R_ret, p_ret)


class HermiteCurve(object):
    def __init__(self, start_pos, start_vel, end_pos, end_vel):
        self._p1 = start_pos
        self._v1 = start_vel
        self._p2 = end_pos
        self._v2 = end_vel

    def evaluate(self, s_in):
        s = np.clip(s_in, 0., 1.)
        return self._p1 * (2 * s**3 - 3 * s**2 + 1) + self._p2 * (
            -2 * s**3 + 3 * s**2) + self._v1 * (
                s**3 - 2 * s**2 + s) + self._v2 * (s**3 - s**2)

    def evaluate_first_derivative(self, s_in):
        s = np.clip(s_in, 0., 1.)

        return self._p1 * (6 * s**2 - 6 * s) + self._p2 * (
            -6 * s**2 + 6 * s) + self._v1 * (
                3 * s**2 - 4 * s + 1) + self._v2 * (3 * s**2 - 2 * s)

    def evaluate_second_derivative(self, s_in):
        s = np.clip(s_in, 0., 1.)

        return self._p1 * (12 * s - 6) + self._p2 * (
            -12 * s + 6) + self._v1 * (6 * s - 4) + self._v2 * (6 * s - 2)


class HermiteCurveVec(object):
    def __init__(self, start_pos, start_vel, end_pos, end_vel):
        self._p1 = np.copy(start_pos)
        self._v1 = np.copy(start_vel)
        self._p2 = np.copy(end_pos)
        self._v2 = np.copy(end_vel)
        self._dim = start_pos.shape[0]

        self._curves = []
        for i in range(self._dim):
            self._curves.append(
                HermiteCurve(start_pos[i], start_vel[i], end_pos[i],
                             end_vel[i]))

    def evaluate(self, s_in):
        return np.array([c.evaluate(s_in) for c in self._curves])

    def evaluate_first_derivative(self, s_in):
        return np.array(
            [c.evaluate_first_derivative(s_in) for c in self._curves])

    def evaluate_second_derivative(self, s_in):
        return np.array(
            [c.evaluate_second_derivative(s_in) for c in self._curves])


class HermiteCurveQuat(object):
    def __init__(self, quat_start, ang_vel_start, quat_end, ang_vel_end):
        self._qa = R.from_quat(quat_start)
        self._omega_a = np.copy(ang_vel_start)
        self._qb = R.from_quat(quat_end)
        self._omega_b = np.copy(ang_vel_end)

        # Initialize Data Structures
        self._q0 = R.from_quat(quat_start)

        if np.linalg.norm(ang_vel_start) < 1e-6:
            self._q1 = R.from_quat(quat_start) * R.from_quat([0., 0., 0., 1.])
        else:
            self._q1 = R.from_quat(quat_start) * R.from_rotvec(
                (np.linalg.norm(ang_vel_start) / 3.0) *
                (ang_vel_start / np.linalg.norm(ang_vel_start)))

        if np.linalg.norm(ang_vel_end) < 1e-6:
            self._q2 = R.from_quat(quat_end) * R.from_quat([0., 0., 0., 1.])
        else:
            self._q2 = R.from_quat(quat_end) * R.from_rotvec(
                (np.linalg.norm(ang_vel_end) / 3.0) *
                (ang_vel_end / np.linalg.norm(ang_vel_end)))

        self._q3 = R.from_quat(quat_end)

        self._omega_1aa = self._q1 * self._q0.inv()
        self._omega_2aa = self._q2 * self._q1.inv()
        self._omega_3aa = self._q3 * self._q2.inv()

        self._omega_1 = self._omega_1aa.as_rotvec()
        self._omega_2 = self._omega_2aa.as_rotvec()
        self._omega_3 = self._omega_3aa.as_rotvec()

    def _compute_basis(self, s_in):
        s = np.clip(s_in, 0., 1.)

        self._b1 = 1 - (1 - s)**3
        self._b2 = 3 * s**2 - 2 * s**3
        self._b3 = s**3
        self._bdot1 = 3 * (1 - s)**2
        self._bdot2 = 6 * s - 6 * s**2
        self._bdot3 = 3 * s**2
        self._bddot1 = -6 * (1 - s)
        self._bddot2 = 6 - 12 * s
        self._bddot3 = 6 * s

    def evaluate(self, s_in):
        s = np.clip(s_in, 0., 1.)
        self._compute_basis(s)

        if np.linalg.norm(self._omega_1) > 1e-5:
            qtmp1 = R.from_rotvec(
                (np.linalg.norm(self._omega_1) * self._b1) *
                (self._omega_1 / np.linalg.norm(self._omega_1)))
        else:
            qtmp1 = R.from_quat([0., 0., 0., 1.])
        if np.linalg.norm(self._omega_2) > 1e-5:
            qtmp2 = R.from_rotvec(
                (np.linalg.norm(self._omega_2) * self._b2) *
                (self._omega_2 / np.linalg.norm(self._omega_2)))
        else:
            qtmp2 = R.from_quat([0., 0., 0., 1.])
        if np.linalg.norm(self._omega_3) > 1e-5:
            qtmp3 = R.from_rotvec(
                (np.linalg.norm(self._omega_3) * self._b3) *
                (self._omega_3 / np.linalg.norm(self._omega_3)))
        else:
            qtmp3 = R.from_quat([0., 0., 0., 1.])

        return (qtmp3 * qtmp2 * qtmp1 * self._q0).as_quat()

    def evaluate_ang_vel(self, s_in):
        s = np.clip(s_in, 0., 1.)
        self._compute_basis(s)

        return self._omega_1 * self._bdot1 + self._omega_2 * self._bdot2 + self._omega_3 * self._bdot3

    def evaluate_ang_acc(self, s_in):
        s = np.clip(s_in, 0., 1.)
        self._compute_basis(s)

        return self._omega_1 * self._bddot1 + self._omega_2 * self._bddot2 + self._omega_3 * self._bddot3
