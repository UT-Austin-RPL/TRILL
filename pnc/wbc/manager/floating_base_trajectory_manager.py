import numpy as np
from scipy.spatial.transform import Rotation as R

from util import geom
from util import interpolation


class FloatingBaseTrajectoryManager(object):
    def __init__(self, com_task, base_ori_task, robot):
        self._com_task = com_task
        self._base_ori_task = base_ori_task
        self._robot = robot

        self._start_time = 0.0
        self._duration = 0.0
        self._ini_com_pos, self._target_com_pos = np.zeros(3), np.zeros(3)
        self._ini_base_quat, self._target_base_quat = np.zeros(4), np.zeros(4)

        self._amp = np.zeros(3)
        self._freq = np.zeros(3)

        self._b_swaying = False

    @property
    def b_swaying(self):
        return self._b_swaying

    @b_swaying.setter
    def b_swaying(self, value):
        self._b_swaying = value

    def initialize_floating_base_interpolation_trajectory(
        self, start_time, duration, target_com_pos, target_base_quat
    ):
        self._start_time = start_time
        self._duration = duration

        self._ini_com_pos = self._robot.get_com_pos()
        self._target_com_pos = target_com_pos

        self._ini_base_quat = geom.rot_to_quat(
            self._robot.get_link_iso(self._base_ori_task.target_id)[0:3, 0:3]
        )
        self._target_base_quat = target_base_quat
        # self._quat_error= (R.from_quat(self._target_base_quat) *
        # R.from_quat(self._ini_base_quat).inv()).as_quat() # Sign flipped
        self._quat_error = R.from_matrix(
            np.dot(
                R.from_quat(self._target_base_quat).as_matrix(),
                R.from_quat(self._ini_base_quat).as_matrix().transpose(),
            )
        ).as_quat()
        self._exp_error = geom.quat_to_exp(self._quat_error)

        # np.set_printoptions(precision=5)
        # print("ini com: ", self._ini_com_pos)
        # print("end com: ", self._target_com_pos)
        # print("ini quat: ", self._ini_base_quat)
        # print("end quat: ", self._target_base_quat)

    def initialize_floating_base_swaying_trajectory(self, start_time, amp, freq):
        self._start_time = start_time
        self._amp = np.copy(amp)
        self._freq = np.copy(freq)

        self._ini_com_pos = self._robot.get_com_pos()
        self._ini_base_quat = geom.rot_to_quat(
            self._robot.get_link_iso(self._base_ori_task.target_id)[0:3, 0:3]
        )
        self._target_base_quat = np.copy(self._ini_base_quat)

    def update_floating_base_desired(self, current_time):
        com_pos_des, com_vel_des, com_acc_des = np.zeros(3), np.zeros(3), np.zeros(3)
        if self._b_swaying:
            com_pos_des, com_vel_des, com_acc_des = geom.get_sinusoid_trajectory(
                self._start_time, self._ini_com_pos, self._amp, self._freq, current_time
            )
        else:
            for i in range(3):
                com_pos_des[i] = interpolation.smooth_changing(
                    self._ini_com_pos[i],
                    self._target_com_pos[i],
                    self._duration,
                    current_time - self._start_time,
                )
                com_vel_des[i] = interpolation.smooth_changing_vel(
                    self._ini_com_pos[i],
                    self._target_com_pos[i],
                    self._duration,
                    current_time - self._start_time,
                )
                com_acc_des[i] = interpolation.smooth_changing_acc(
                    self._ini_com_pos[i],
                    self._target_com_pos[i],
                    self._duration,
                    current_time - self._start_time,
                )

        self._com_task.update_desired(com_pos_des, com_vel_des, com_acc_des)

        scaled_t = interpolation.smooth_changing(
            0, 1, self._duration, current_time - self._start_time
        )
        scaled_tdot = interpolation.smooth_changing_vel(
            0, 1, self._duration, current_time - self._start_time
        )
        scaled_tddot = interpolation.smooth_changing_acc(
            0, 1, self._duration, current_time - self._start_time
        )

        exp_inc = self._exp_error * scaled_t
        quat_inc = geom.exp_to_quat(exp_inc)

        # TODO (Check this again)
        # base_quat_des = R.from_matrix(
        # np.dot(
        # R.from_quat(quat_inc).as_matrix(),
        # R.from_quat(self._ini_base_quat).as_matrix())).as_quat()
        base_quat_des = R.from_matrix(
            np.dot(
                R.from_quat(self._ini_base_quat).as_matrix(),
                R.from_quat(quat_inc).as_matrix(),
            )
        ).as_quat()
        base_angvel_des = self._exp_error * scaled_tdot
        base_angacc_des = self._exp_error * scaled_tddot

        self._base_ori_task.update_desired(
            base_quat_des, base_angvel_des, base_angacc_des
        )
