import copy

import numpy as np

from pnc.dcm import Footstep, interpolate
from util import geom
from util import interpolation


class FootTrajectoryManager(object):
    """
    Foot SE(3) Trajectory Manager
    -----------------------------
    Usage:
        use_current or
        initialize_swing_foot_trajectory --> update_swing_foot_desired
    """
    def __init__(self, pos_task, ori_task, robot):
        self._pos_task = pos_task
        self._ori_task = ori_task
        self._robot = robot

        self._swing_start_time = 0.
        self._swing_duration = 0.

        self._swing_init_foot = Footstep()
        self._swing_mid_foot = Footstep()
        self._swing_land_foot = Footstep()

        self._pos_traj_init_to_mid = None
        self._pos_traj_mid_to_end = None
        self._quat_hermite_curve = None

        assert self._pos_task.target_id == self._ori_task.target_id
        self._target_id = self._pos_task.target_id

        # Attribute
        self._swing_height = 0.05

    def use_current(self):
        foot_iso = self._robot.get_link_iso(self._target_id)
        foot_vel = self._robot.get_link_vel(self._target_id)

        foot_pos_des = foot_iso[0:3, 3]
        foot_lin_vel_des = foot_vel[3:6]
        self._pos_task.update_desired(foot_pos_des, foot_lin_vel_des,
                                      np.zeros(3))

        foot_rot_des = geom.rot_to_quat(foot_iso[0:3, 0:3])
        foot_ang_vel_des = foot_vel[0:3]
        self._ori_task.update_desired(foot_rot_des, foot_ang_vel_des,
                                      np.zeros(3))

    def initialize_swing_foot_trajectory(self, start_time, swing_duration,
                                         landing_foot):
        self._swing_start_time = start_time
        self._swing_duration = swing_duration
        self._swing_land_foot = copy.deepcopy(landing_foot)

        self._swing_init_foot.iso = np.copy(
            self._robot.get_link_iso(self._target_id))
        self._swing_init_foot.side = landing_foot.side
        self._swing_mid_foot = interpolate(self._swing_init_foot, landing_foot,
                                           0.5)

        # compute midfoot boundary conditions
        mid_swing_local_foot_pos = np.array([0., 0., self._swing_height])
        mid_swing_pos = self._swing_mid_foot.pos + np.dot(
            self._swing_mid_foot.rot, mid_swing_local_foot_pos)
        mid_swing_vel = (self._swing_land_foot.pos -
                         self._swing_init_foot.pos) / self._swing_duration

        # construct trajectories
        self._pos_traj_init_to_mid = interpolation.HermiteCurveVec(
            self._swing_init_foot.pos, np.zeros(3), mid_swing_pos,
            mid_swing_vel)
        self._pos_traj_mid_to_end = interpolation.HermiteCurveVec(
            mid_swing_pos, mid_swing_vel, self._swing_land_foot.pos,
            np.zeros(3))
        self._quat_hermite_curve = interpolation.HermiteCurveQuat(
            self._swing_init_foot.quat, np.zeros(3),
            self._swing_land_foot.quat, np.zeros(3))

    def update_swing_foot_desired(self, curr_time):
        s = (curr_time - self._swing_start_time) / self._swing_duration

        if s <= 0.5:
            s = 2.0 * s
            foot_pos_des = self._pos_traj_init_to_mid.evaluate(s)
            foot_vel_des = self._pos_traj_init_to_mid.evaluate_first_derivative(
                s)
            foot_acc_des = self._pos_traj_init_to_mid.evaluate_second_derivative(
                s)
        else:
            s = 2.0 * (s - 0.5)
            foot_pos_des = self._pos_traj_mid_to_end.evaluate(s)
            foot_vel_des = self._pos_traj_mid_to_end.evaluate_first_derivative(
                s)
            foot_acc_des = self._pos_traj_mid_to_end.evaluate_second_derivative(
                s)

        foot_quat_des = self._quat_hermite_curve.evaluate(s)
        foot_ang_vel_des = self._quat_hermite_curve.evaluate_ang_vel(s)
        foot_ang_acc_des = self._quat_hermite_curve.evaluate_ang_acc(s)

        self._pos_task.update_desired(foot_pos_des, foot_vel_des, foot_acc_des)
        self._ori_task.update_desired(foot_quat_des, foot_ang_vel_des,
                                      foot_ang_acc_des)

    @property
    def swing_height(self):
        return self._swing_height

    @swing_height.setter
    def swing_height(self, val):
        self._swing_height = val
