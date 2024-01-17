import copy
import math
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R

from pnc.dcm.footstep import Footstep
from pnc.dcm.footstep import interpolate


class DCMTransferType(object):
    INI = 0
    MID = 1


class DCMTrajectoryManager(object):
    def __init__(self, dcm_planner, com_task, base_ori_task, robot, lfoot_id, rfoot_id):
        self._dcm_planner = dcm_planner
        self._com_task = com_task
        self._base_ori_task = base_ori_task
        self._robot = robot
        self._lfoot_id = lfoot_id
        self._rfoot_id = rfoot_id

        self._des_com_pos = np.zeros(3)
        self._des_com_vel = np.zeros(3)
        self._des_com_acc = np.zeros(3)

        self._des_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._des_ang_vel = np.zeros(3)
        self._des_ang_acc = np.zeros(3)

        self._reset_step_idx()

        self._robot_side = Footstep.RIGHT_SIDE

        self._footstep_list = []
        self._footstep_preview_list = []

        self._lf_stance = Footstep()
        self._rf_stance = Footstep()
        self._mf_stance = Footstep()

        # Attributes
        self._nominal_com_height = 1.015
        self._t_additional_init_transfer = 0.0
        self._t_contact_transition = 0.45
        self._t_swing = 1.0
        self._percentage_settle = 0.99
        self._alpha_ds = 0.5
        self._nominal_footwidth = 0.27
        self._nominal_forward_step = 0.25
        self._nominal_backward_step = -0.25
        self._nominal_turn_radians = np.pi / 4.0
        self._nominal_strafe_distance = 0.125

        self._set_temporal_params()

    def compute_ini_contact_transfer_time(self):
        return (
            self._t_additional_init_transfer
            + self._t_ds
            + (1 - self._alpha_ds) * self._t_ds
        )

    def compute_mid_step_contact_transfer_time(self):
        return self._t_ds

    def compute_final_contact_transfer_time(self):
        return self._t_ds + self._dcm_planner.compute_settling_time()

    def compute_swing_time(self):
        return self._t_ss

    def compute_rf_z_ramp_up_time(self):
        return self._alpha_ds * self._t_ds

    def compute_rf_z_ramp_down_time(self):
        return (1.0 - self._alpha_ds) * self._t_ds

    def _update_starting_stance(self):
        self._lf_stance.iso = self._robot.get_link_iso(self._lfoot_id)
        self._lf_stance.side = Footstep.LEFT_SIDE
        self._rf_stance.iso = self._robot.get_link_iso(self._rfoot_id)
        self._rf_stance.side = Footstep.RIGHT_SIDE
        self._mf_stance = interpolate(self._lf_stance, self._rf_stance, 0.5)

    # TODO : Put initial angular velocity
    def initialize(
        self, t_start, transfer_type, quat_start, dcm_pos_start, dcm_vel_start
    ):
        self._update_starting_stance()
        self._update_footstep_preview()

        self._dcm_planner.robot_mass = self._robot.total_mass
        self._dcm_planner.z_vrp = self._nominal_com_height
        self._dcm_planner.t_start = t_start
        self._dcm_planner.ini_quat = quat_start

        if transfer_type == DCMTransferType.INI:
            self._dcm_planner.t_transfer = self._t_transfer_ini
        elif transfer_type == DCMTransferType.MID:
            self._dcm_planner.t_transfer = self._t_transfer_mid
        else:
            raise ValueError("Wrong DCMTransferType")

        self._dcm_planner.initialize(
            self._footstep_preview_list,
            self._lf_stance,
            self._rf_stance,
            dcm_pos_start,
            dcm_vel_start,
        )

    def save_trajectory(self, file_name):
        t_start = self._dcm_planner.t_start
        t_end = t_start + self._dcm_planner.t_end
        t_step = 0.01
        n_eval = math.floor((t_end - t_start) / t_step)

        data = dict()

        # Temporal Info
        data["temporal_parameters"] = dict()
        data["temporal_parameters"]["initial_time"] = t_start
        data["temporal_parameters"]["final_time"] = t_end
        data["temporal_parameters"]["time_step"] = t_step
        data["temporal_parameters"]["t_ds"] = self._dcm_planner.t_ds
        data["temporal_parameters"]["t_ss"] = self._dcm_planner.t_ss
        data["temporal_parameters"]["t_transfer"] = self._dcm_planner.t_transfer

        # Contact Info
        data["contact"] = dict()
        data["contact"]["curr_right_foot"] = dict()
        data["contact"]["curr_left_foot"] = dict()
        data["contact"]["right_foot"] = dict()
        data["contact"]["left_foot"] = dict()
        data["contact"]["curr_right_foot"]["pos"] = np.copy(self._rf_stance.pos)
        data["contact"]["curr_right_foot"]["ori"] = np.copy(self._rf_stance.quat)
        data["contact"]["curr_left_foot"]["pos"] = np.copy(self._lf_stance.pos)
        data["contact"]["curr_left_foot"]["ori"] = np.copy(self._lf_stance.quat)

        rfoot_pos, rfoot_quat, lfoot_pos, lfoot_quat = [], [], [], []
        for i in range(len(self._footstep_list)):
            if self._footstep_list[i].side == Footstep.RIGHT_SIDE:
                rfoot_pos.append(self._footstep_list[i].pos)
                rfoot_quat.append(self._footstep_list[i].quat)
            else:
                lfoot_pos.append(self._footstep_list[i].pos)
                lfoot_quat.append(self._footstep_list[i].quat)
        data["contact"]["right_foot"]["pos"] = rfoot_pos
        data["contact"]["right_foot"]["ori"] = rfoot_quat
        data["contact"]["left_foot"]["pos"] = lfoot_pos
        data["contact"]["left_foot"]["ori"] = lfoot_quat

        # Ref Trajectory
        com_pos_ref = np.zeros((n_eval, 3))
        com_vel_ref = np.zeros((n_eval, 3))
        base_ori_ref = np.zeros((n_eval, 4))
        t_traj = np.zeros((n_eval, 1))

        t = t_start
        for i in range(n_eval):
            t_traj[i, 0] = t
            com_pos_ref[i, :] = self._dcm_planner.compute_reference_com_pos(t)
            com_vel_ref[i, :] = self._dcm_planner.compute_reference_com_vel(t)
            base_ori_ref[i, :], _, _ = self._dcm_planner.compute_reference_base_ori(t)
            t += t_step

        data["reference"] = dict()
        data["reference"]["com_pos"] = com_pos_ref
        data["reference"]["com_vel"] = com_vel_ref
        data["reference"]["base_ori"] = base_ori_ref
        data["reference"]["time"] = t_traj

        if not os.path.exists("data"):
            os.makedirs("data")

        file = open("data/{}_th_dcm_planning.pkl".format(file_name), "ab")
        pickle.dump(data, file)
        file.close()

    def update_floating_base_task_desired(self, curr_time):
        self._des_com_pos = self._dcm_planner.compute_reference_com_pos(curr_time)
        self._des_com_vel = self._dcm_planner.compute_reference_com_vel(curr_time)
        self._des_com_acc = np.zeros(3)  # TODO : Compute com_acc
        (
            self._des_quat,
            self._des_ang_vel,
            self._des_ang_acc,
        ) = self._dcm_planner.compute_reference_base_ori(curr_time)

        self._com_task.update_desired(
            self._des_com_pos, self._des_com_vel, self._des_com_acc
        )
        self._base_ori_task.update_desired(
            self._des_quat, self._des_ang_vel, self._des_ang_acc
        )

    def next_step_side(self):
        if len(self._footstep_list) > 0 and self._curr_footstep_idx < len(
            self._footstep_list
        ):
            return True, self._footstep_list[self._curr_footstep_idx].side
        else:
            return False, None

    def no_reaming_steps(self):
        if self._curr_footstep_idx >= len(self._footstep_list):
            return True
        else:
            return False

    def increment_step_idx(self):
        self._curr_footstep_idx += 1

    def walk_in_x(self, goal_x):
        self._reset_idx_and_clear_footstep_list()
        self._update_starting_stance()
        curr_x = self._mf_stance.iso[0, 3]
        nstep = int(np.abs(goal_x - curr_x) // self._nominal_forward_step)
        xlen = (goal_x - curr_x) / nstep
        self._populate_walk_forward(nstep, xlen)
        self._alternate_leg()

    def walk_in_y(self, goal_y):
        self._reset_idx_and_clear_footstep_list()
        self._update_starting_stance()
        curr_y = self._mf_stance.iso[1, 3]
        nstep = int(np.abs(goal_y - curr_y) // self._nominal_strafe_distance)
        ylen = (goal_y - curr_y) / nstep
        self._populate_strafe(nstep, ylen)

    def walk_in_place(self):
        self._reset_idx_and_clear_footstep_list()
        self._populate_step_in_place(1, self._robot_side)
        self._alternate_leg()

    def walk_forward(self):
        self._reset_idx_and_clear_footstep_list()
        self._populate_walk_forward(1, self._nominal_forward_step)
        self._alternate_leg()

    def walk_backward(self):
        self._reset_idx_and_clear_footstep_list()
        self._populate_walk_forward(1, self._nominal_backward_step)
        self._alternate_leg()

    def strafe_left(self):
        self._reset_idx_and_clear_footstep_list()
        self._populate_strafe(1, self._nominal_strafe_distance)

    def strafe_right(self):
        self._reset_idx_and_clear_footstep_list()
        self._populate_strafe(1, -self._nominal_strafe_distance)

    def turn_left(self):
        self._reset_idx_and_clear_footstep_list()
        self._populate_turn(1, self._nominal_turn_radians)

    def turn_right(self):
        self._reset_idx_and_clear_footstep_list()
        self._populate_turn(1, -self._nominal_turn_radians)

    def _populate_step_in_place(self, num_step, robot_side_first):
        self._update_starting_stance()

        lf_stance = copy.deepcopy(self._lf_stance)
        rf_stance = copy.deepcopy(self._rf_stance)
        mf_stance = copy.deepcopy(self._mf_stance)
        robot_side = robot_side_first
        for i in range(num_step):
            if robot_side == Footstep.LEFT_SIDE:
                lf_stance.pos = mf_stance.pos + np.dot(
                    mf_stance.rot, np.array([0.0, self._nominal_footwidth / 2.0, 0.0])
                )
                lf_stance.rot = np.copy(mf_stance.rot)
                # lf_stance.side = Footstep.LEFT_SIDE # TODO : Do I need this?
                self._footstep_list.append(copy.deepcopy(lf_stance))
                robot_side = Footstep.RIGHT_SIDE

            else:
                rf_stance.pos = mf_stance.pos + np.dot(
                    mf_stance.rot, np.array([0.0, -self._nominal_footwidth / 2.0, 0.0])
                )
                rf_stance.rot = np.copy(mf_stance.rot)
                # rf_stance.side = Footstep.RIGHT_SIDE # TODO : Do I need this?
                self._footstep_list.append(copy.deepcopy(rf_stance))
                robot_side = Footstep.LEFT_SIDE

    def _populate_walk_forward(self, num_steps, forward_distance):
        self._update_starting_stance()

        new_stance = Footstep()
        mf_stance = copy.deepcopy(self._mf_stance)

        robot_side = Footstep.LEFT_SIDE
        for i in range(num_steps):
            if robot_side == Footstep.LEFT_SIDE:
                translate = np.array(
                    [(i + 1) * forward_distance, self._nominal_footwidth / 2.0, 0.0]
                )
                new_stance.pos = mf_stance.pos + np.dot(mf_stance.rot, translate)
                new_stance.rot = np.copy(mf_stance.rot)
                new_stance.side = Footstep.LEFT_SIDE
                robot_side = Footstep.RIGHT_SIDE
            else:
                translate = np.array(
                    [(i + 1) * forward_distance, -self._nominal_footwidth / 2.0, 0.0]
                )
                new_stance.pos = mf_stance.pos + np.dot(mf_stance.rot, translate)
                new_stance.rot = np.copy(mf_stance.rot)
                new_stance.side = Footstep.RIGHT_SIDE
                robot_side = Footstep.LEFT_SIDE
            self._footstep_list.append(copy.deepcopy(new_stance))

        # Add additional step forward to square the feet
        if robot_side == Footstep.LEFT_SIDE:
            translate = np.array(
                [num_steps * forward_distance, self._nominal_footwidth / 2.0, 0.0]
            )
            new_stance.pos = mf_stance.pos + np.dot(mf_stance.rot, translate)
            new_stance.rot = mf_stance.rot
            new_stance.side = Footstep.LEFT_SIDE
        else:
            translate = np.array(
                [num_steps * forward_distance, -self._nominal_footwidth / 2.0, 0.0]
            )
            new_stance.pos = mf_stance.pos + np.dot(mf_stance.rot, translate)
            new_stance.rot = mf_stance.rot
            new_stance.side = Footstep.RIGHT_SIDE
        self._footstep_list.append(copy.deepcopy(new_stance))

    def _populate_turn(self, num_times, turn_radians_per_step):
        self._update_starting_stance()

        foot_rotation = R.from_rotvec([0.0, 0.0, turn_radians_per_step]).as_matrix()

        lf_stance = Footstep()
        rf_stance = Footstep()
        mf_stance = copy.deepcopy(self._mf_stance)
        mf_stance_rotated = copy.deepcopy(self._mf_stance)

        for i in range(num_times):
            mf_stance_rotated.pos = np.copy(mf_stance.pos)
            mf_stance_rotated.rot = np.dot(foot_rotation, mf_stance.rot)

            lf_stance.pos = mf_stance_rotated.pos + np.dot(
                mf_stance_rotated.rot,
                np.array([0.0, self._nominal_footwidth / 2.0, 0.0]),
            )
            lf_stance.rot = np.copy(mf_stance_rotated.rot)
            lf_stance.side = Footstep.LEFT_SIDE

            rf_stance.pos = mf_stance_rotated.pos + np.dot(
                mf_stance_rotated.rot,
                np.array([0.0, -self._nominal_footwidth / 2.0, 0.0]),
            )
            rf_stance.rot = np.copy(mf_stance_rotated.rot)
            rf_stance.side = Footstep.RIGHT_SIDE

            if turn_radians_per_step > 0.0:
                self._footstep_list.append(copy.deepcopy(lf_stance))
                self._footstep_list.append(copy.deepcopy(rf_stance))
            else:
                self._footstep_list.append(copy.deepcopy(rf_stance))
                self._footstep_list.append(copy.deepcopy(lf_stance))
            mf_stance = copy.deepcopy(mf_stance_rotated)

    def _populate_strafe(self, num_times, strafe_distance):
        self._update_starting_stance()

        lf_stance = Footstep()
        rf_stance = Footstep()
        mf_stance = copy.deepcopy(self._mf_stance)
        mf_stance_translated = copy.deepcopy(self._mf_stance)

        for i in range(num_times):
            mf_stance_translated.pos = mf_stance.pos + np.dot(
                mf_stance.rot, np.array([0.0, strafe_distance, 0.0])
            )
            mf_stance_translated.rot = mf_stance.rot

            lf_stance.pos = mf_stance_translated.pos + np.dot(
                mf_stance_translated.rot,
                np.array([0.0, self._nominal_footwidth / 2.0, 0.0]),
            )
            lf_stance.rot = mf_stance_translated.rot
            lf_stance.side = Footstep.LEFT_SIDE

            rf_stance.pos = mf_stance_translated.pos + np.dot(
                mf_stance_translated.rot,
                np.array([0.0, -self._nominal_footwidth / 2.0, 0.0]),
            )
            rf_stance.rot = mf_stance_translated.rot
            rf_stance.side = Footstep.RIGHT_SIDE

            if strafe_distance > 0:
                self._footstep_list.append(copy.deepcopy(lf_stance))
                self._footstep_list.append(copy.deepcopy(rf_stance))
            else:
                self._footstep_list.append(copy.deepcopy(rf_stance))
                self._footstep_list.append(copy.deepcopy(lf_stance))
            mf_stance = copy.deepcopy(mf_stance_translated)

    def _alternate_leg(self):
        if self._robot_side == Footstep.LEFT_SIDE:
            self._robot_side = Footstep.RIGHT_SIDE
        else:
            self._robot_side = Footstep.LEFT_SIDE

    def _reset_idx_and_clear_footstep_list(self):
        self._reset_step_idx()
        self._footstep_list = []

    def _update_footstep_preview(self, max_footsteps_to_preview=40):
        self._footstep_preview_list = []
        for i in range(max_footsteps_to_preview):
            if (i + self._curr_footstep_idx) < len(self._footstep_list):
                self._footstep_preview_list.append(
                    self._footstep_list[i + self._curr_footstep_idx]
                )
            else:
                break

    def _reset_step_idx(self):
        self._curr_footstep_idx = 0

    def _set_temporal_params(self):
        self._t_ds = self._t_contact_transition
        self._t_ss = self._t_swing
        self._t_transfer_ini = self._t_additional_init_transfer
        self._t_transfer_mid = (self._alpha_ds - 1.0) * self._t_ds

        self._dcm_planner.t_transfer = self._t_transfer_ini
        self._dcm_planner.t_ds = self._t_ds
        self._dcm_planner.t_ss = self._t_ss
        self._dcm_planner.percentage_settle = self._percentage_settle
        self._dcm_planner.alpha_ds = self._alpha_ds

    @property
    def nominal_com_height(self):
        return self._nominal_com_height

    @nominal_com_height.setter
    def nominal_com_height(self, value):
        self._nominal_com_height = value

    @property
    def t_additional_init_transfer(self):
        return self._t_additional_init_transfer

    @t_additional_init_transfer.setter
    def t_additional_init_transfer(self, value):
        self._t_additional_init_transfer = value
        self._t_transfer_ini = self._t_additional_init_transfer
        self._dcm_planner.t_transfer = self._t_transfer_ini

    @property
    def t_contact_transition(self):
        return self._t_contact_transition

    @t_contact_transition.setter
    def t_contact_transition(self, value):
        self._t_contact_transition = value
        self._t_ds = self._t_contact_transition
        self._t_transfer_mid = (self._alpha_ds - 1.0) * self._t_ds
        self._dcm_planner.t_ds = self._t_ds

    @property
    def t_swing(self):
        return self._t_swing

    @t_swing.setter
    def t_swing(self, value):
        self._t_swing = value
        self._t_ss = self._t_swing
        self._dcm_planner.t_ss = self._t_ss

    @property
    def percentage_settle(self):
        return self._percentage_settle

    @percentage_settle.setter
    def percentage_settle(self, value):
        self._percentage_settle = value
        self._dcm_planner.percentage_settle = self._percentage_settle

    @property
    def alpha_ds(self):
        return self._alpha_ds

    @alpha_ds.setter
    def alpha_ds(self, value):
        self._alpha_ds = value
        self._t_transfer_mid = (self._alpha_ds - 1.0) * self._t_ds
        self._dcm_planner.alpha_ds = self._alpha_ds

    @property
    def nominal_footwidth(self):
        return self._nominal_footwidth

    @nominal_footwidth.setter
    def nominal_footwidth(self, value):
        self._nominal_footwidth = value

    @property
    def nominal_forward_step(self):
        return self._nominal_forward_step

    @nominal_forward_step.setter
    def nominal_forward_step(self, value):
        self._nominal_forward_step = value

    @property
    def nominal_backward_step(self):
        return self._nominal_backward_step

    @nominal_backward_step.setter
    def nominal_backward_step(self, value):
        self._nominal_backward_step = value

    @property
    def nominal_turn_radians(self):
        return self._nominal_turn_radians

    @nominal_turn_radians.setter
    def nominal_turn_radians(self, value):
        self._nominal_turn_radians = value

    @property
    def nominal_strafe_distance(self):
        return self._nominal_strafe_distance

    @nominal_strafe_distance.setter
    def nominal_strafe_distance(self, value):
        self._nominal_strafe_distance = value

    @property
    def nominal_turn_radians(self):
        return self._nominal_turn_radians

    @nominal_turn_radians.setter
    def nominal_turn_radians(self, value):
        self._nominal_turn_radians = value

    @property
    def footstep_list(self):
        return self._footstep_list

    @property
    def curr_footstep_idx(self):
        return self._curr_footstep_idx
