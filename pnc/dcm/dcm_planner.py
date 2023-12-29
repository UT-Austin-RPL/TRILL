import copy

import numpy as np

from util import interpolation
from .footstep import Footstep, interpolate


class VRPType(object):
    RF_SWING = 0
    LF_SWING = 1
    TRANSFER = 2
    END = 3


class DCMPlanner(object):
    def __init__(self):
        self._ini_lf_stance = Footstep()
        self._ini_rf_stance = Footstep()
        self._ini_dcm_pos = np.zeros(3)
        self._ini_dcm_vel = np.zeros(3)

        self._vrp_list = []  # vrp
        self._vrp_type_list = []

        # Attributes
        self._t_transfer = 0.1
        self._t_ds = 0.05
        self._t_ss = 0.3
        self._percentage_settle = 0.99
        self._alpha_ds = 0.5

        self._t_start = 0.0
        self._t_end = 0.0
        self._dt = 1e-3

        self._z_vrp = 0.75  # com height
        self._b = np.sqrt(self._z_vrp / 9.81)
        self._robot_mass = 50
        self._ini_quat = np.array([0.0, 0.0, 0.0, 1])

    def initialize(
        self,
        input_footstep_list,
        left_footstance,
        right_footstance,
        ini_dcm,
        ini_dcm_vel,
    ):
        self._vrp_list = []
        self._vrp_type_list = []

        self._ini_lf_stance = left_footstance
        self._ini_rf_stance = right_footstance
        self._ini_dcm_pos = ini_dcm
        self._ini_dcm_vel = ini_dcm_vel

        self._vrp_list.append(np.copy(ini_dcm))
        self._vrp_type_list.append(VRPType.TRANSFER)

        self._footstep_list = input_footstep_list
        if input_footstep_list[0].side == Footstep.LEFT_SIDE:
            ini_footstance = copy.deepcopy(self._ini_rf_stance)
        else:
            ini_footstance = copy.deepcopy(self._ini_lf_stance)

        curr_stance_vrp = (
            np.dot(ini_footstance.rot, np.array([0.0, 0.0, self._z_vrp]))
            + ini_footstance.pos
        )
        left_stance_vrp = np.copy(curr_stance_vrp)
        right_stance_vrp = np.copy(curr_stance_vrp)

        self._vrp_list.append(np.copy(curr_stance_vrp))

        prev_side = ini_footstance.side

        for i in range(len(input_footstep_list)):
            curr_vrp = np.array([0.0, 0.0, self._z_vrp])
            curr_vrp = (
                np.dot(input_footstep_list[i].rot, curr_vrp)
                + input_footstep_list[i].pos
            )

            if input_footstep_list[i].side == Footstep.LEFT_SIDE:
                curr_stance_vrp = np.copy(right_stance_vrp)
            else:
                curr_stance_vrp = np.copy(left_stance_vrp)

            if i == len(input_footstep_list) - 1:
                curr_vrp = 0.5 * (curr_vrp + curr_stance_vrp)

            if input_footstep_list[i].side == prev_side:
                self._vrp_type_list.append(VRPType.TRANSFER)
                self._vrp_list.append(curr_stance_vrp)
            else:
                if input_footstep_list[i].side == Footstep.LEFT_SIDE:
                    left_stance_vrp = np.copy(curr_vrp)
                else:
                    right_stance_vrp = np.copy(curr_vrp)

            if input_footstep_list[i].side == Footstep.LEFT_SIDE:
                self._vrp_type_list.append(VRPType.LF_SWING)
            else:
                self._vrp_type_list.append(VRPType.RF_SWING)

            self._vrp_list.append(curr_vrp)
            prev_side = input_footstep_list[i].side

        self._vrp_type_list.append(VRPType.END)

        self._compute_dcm_trajectory()

    def compute_reference_com_pos(self, t):
        time = np.clip(t - self._t_start, 0.0, self._t_end)
        idx = int(time / self._dt)
        return self._ref_com_pos[idx]

    def compute_reference_com_vel(self, t):
        if t < self._t_start:
            return np.zeros(3)
        time = np.clip(t - self._t_start, 0.0, self._t_end)
        idx = int(time / self._dt)
        return self._ref_com_vel[idx]

    def compute_reference_base_ori(self, t):
        time = np.clip(t - self._t_start, 0.0, self._t_end)
        step_idx = self._compute_step_idx(time)
        t_traj_start = self._compute_t_step_start(step_idx)
        t_traj_end = self._compute_t_step(step_idx)
        traj_duration = t_traj_end - t_traj_start
        time_query = np.clip(time, t_traj_start, t_traj_end)
        s = (time_query - t_traj_start) / traj_duration

        b_swinging, t_swing_start, t_swing_end = self._compute_t_swing_start_end(
            step_idx
        )
        if b_swinging:
            time_query = np.clip(time, t_swing_start, t_swing_end)
            traj_duration = t_swing_end - t_swing_start
            s = (time_query - t_swing_start) / traj_duration

        des_quat = self._base_quat_curves[step_idx].evaluate(s)
        des_ang_vel = self._base_quat_curves[step_idx].evaluate_ang_vel(s)
        des_ang_acc = self._base_quat_curves[step_idx].evaluate_ang_acc(s)

        return des_quat, des_ang_vel, des_ang_acc

    def _compute_t_swing_start_end(self, step_idx):
        if (
            self._vrp_type_list[step_idx] == VRPType.LF_SWING
            or self._vrp_type_list[step_idx] == VRPType.RF_SWING
        ):
            swing_start_time = self._compute_t_step_start(step_idx) + self._t_ds * (
                1.0 - self._alpha_ds
            )
            swing_end_time = self._compute_t_step_end(step_idx) - (
                self._alpha_ds * self._t_ds
            )
            return True, swing_start_time, swing_end_time
        else:
            return False, None, None

    def _compute_dcm_trajectory(self):
        self._dcm_ini_list = [None] * len(self._vrp_list)
        self._dcm_eos_list = [None] * len(self._vrp_list)
        self._dcm_ini_ds_list = [None] * len(self._vrp_list)
        self._dcm_vel_ini_ds_list = [None] * len(self._vrp_list)
        self._dcm_acc_ini_ds_list = [None] * len(self._vrp_list)
        self._dcm_end_ds_list = [None] * len(self._vrp_list)
        self._dcm_vel_end_ds_list = [None] * len(self._vrp_list)
        self._dcm_acc_end_ds_list = [None] * len(self._vrp_list)

        self._dcm_P = [None] * len(self._vrp_list)
        # self._dcm_minjerk = [None] * len(self._vrp_list)

        # Use backwards recursion to compute the initial and final dcm states.
        # Last element of the DCM end of step list is equal to the last rvrp.
        self._dcm_eos_list[-1] = np.copy(self._vrp_list[-1])
        for i in reversed(range(len(self._dcm_ini_list))):
            t_step = self._compute_t_step(i)
            # compute dcm_ini for step i
            self._dcm_ini_list[i] = self._compute_dcm_ini(
                self._vrp_list[i], t_step, self._dcm_eos_list[i]
            )
            # set dcm_eos for step i-1
            if i > 0:
                self._dcm_eos_list[i - 1] = np.copy(self._dcm_ini_list[i])

        # Find boundary conditions for the polynomial interpolator
        for i in range(len(self._vrp_list)):
            self._dcm_ini_ds_list[i] = self._compute_dcm_ini_ds(
                i, self._alpha_ds * self._t_ds
            )
            self._dcm_vel_ini_ds_list[i] = self._compute_dcm_vel_ini_ds(
                i, self._alpha_ds * self._t_ds
            )
            self._dcm_acc_ini_ds_list[i] = self._compute_dcm_acc_ini_ds(
                i, self._alpha_ds * self._t_ds
            )
            self._dcm_end_ds_list[i] = self._compute_dcm_end_ds(
                i, (1 - self._alpha_ds) * self._t_ds
            )
            self._dcm_vel_end_ds_list[i] = self._compute_dcm_vel_end_ds(
                i, (1 - self._alpha_ds) * self._t_ds
            )
            self._dcm_acc_end_ds_list[i] = self._compute_dcm_acc_end_ds(
                i, (1 - self._alpha_ds) * self._t_ds
            )

        # Recompute first DS polynomial boundary conditions again
        self._dcm_end_ds_list[0] = self._compute_dcm_end_ds(
            0, self._t_transfer + self._alpha_ds * self._t_ds
        )
        self._dcm_vel_end_ds_list[0] = self._compute_dcm_vel_end_ds(
            0, self._t_transfer + self._alpha_ds * self._t_ds
        )
        self._dcm_acc_end_ds_list[0] = self._compute_dcm_acc_end_ds(
            0, self._t_transfer + self._alpha_ds * self._t_ds
        )

        # self._print_boundary_conditions()

        # Compute polynomial interpolator matrix
        for i in range(len(self._vrp_list)):
            ts = self._compute_polynomial_duration(i)
            self._dcm_P[i] = self._compute_polynomial_matrix(
                ts,
                self._dcm_ini_ds_list[i],
                self._dcm_vel_ini_ds_list[i],
                self._dcm_end_ds_list[i],
                self._dcm_vel_end_ds_list[i],
            )

            # self._dcm_minjerk[i] = self._compute_minjerk_curve_vec(
            # self._dcm_ini_ds_list[i], self._dcm_vel_ini_ds_list[i],
            # self._dcm_acc_ini_ds_list[i], self._dcm_end_ds_list[i],
            # self._dcm_vel_end_ds_list[i], self._dcm_acc_end_ds_list[i], ts)

        self._compute_total_trajectory_time()
        self._compute_reference_com_trajectory()
        self._compute_reference_base_ori_trajectory()

    def _compute_reference_base_ori_trajectory(self):
        self._base_quat_curves = []

        prev_lf_stance = self._ini_lf_stance
        prev_rf_stance = self._ini_rf_stance

        curr_base_quat = self._ini_quat

        step_counter = 0
        for i in range(len(self._vrp_type_list)):
            # swing state
            if (
                self._vrp_type_list[i] == VRPType.RF_SWING
                or self._vrp_type_list[i] == VRPType.LF_SWING
            ):
                target_step = self._footstep_list[step_counter]
                if target_step.side == Footstep.LEFT_SIDE:
                    stance_step = prev_rf_stance
                    prev_lf_stance = target_step
                else:
                    stance_step = prev_lf_stance
                    prev_rf_stance = target_step
                # Find the midefeet
                mid_foot_stance = interpolate(stance_step, target_step, 0.5)
                # Create the hermite quaternion curve object
                self._base_quat_curves.append(
                    interpolation.HermiteCurveQuat(
                        curr_base_quat, np.zeros(3), mid_foot_stance.quat, np.zeros(3)
                    )
                )
                # Update the base orientation
                curr_base_quat = np.copy(mid_foot_stance.quat)
                step_counter += 1
            else:
                # TODO : Initiate angular velocity with curr angular velocity
                mid_foot_stance = interpolate(prev_lf_stance, prev_rf_stance, 0.5)
                curr_base_quat = mid_foot_stance.quat
                self._base_quat_curves.append(
                    interpolation.HermiteCurveQuat(
                        curr_base_quat, np.zeros(3), curr_base_quat, np.zeros(3)
                    )
                )

    def _compute_reference_com_trajectory(self):
        self._compute_total_trajectory_time()
        t_local = self._t_start
        t_local_end = self._t_start + self._t_end

        n_local = int(self._t_end / self._dt)

        self._ref_com_pos = [None] * (n_local + 1)
        self._ref_com_vel = [None] * (n_local + 1)

        com_pos = np.copy(self._vrp_list[0])
        com_vel = np.zeros(3)  # TODO: Initialize this from initial com vel
        dcm_cur = np.zeros(3)

        for i in range(n_local + 1):
            t_local = self._t_start + i * self._dt
            dcm_cur = self._compute_ref_dcm(t_local)
            com_vel = self._compute_com_vel(com_pos, dcm_cur)
            com_pos = com_pos + com_vel * self._dt

            self._ref_com_pos[i] = np.copy(com_pos)
            self._ref_com_vel[i] = np.copy(com_vel)

    def _compute_com_vel(self, com_pos, dcm):
        return (-1.0 / self._b) * (com_pos - dcm)

    def _compute_ref_dcm(self, t):
        if t < self._t_start:
            return self._vrp_list[0]
        time = np.clip(t - self._t_start, 0.0, self._t_end)
        step_idx = self._compute_step_idx(time)
        local_time = 0.0
        if time <= self._compute_ds_t_end(step_idx):
            local_time = time - self._compute_ds_t_start(step_idx)
            dcm_out = self._compute_dcm_ds_poly(step_idx, local_time)
        else:
            local_time = time - self._compute_t_step_start(step_idx)
            dcm_out = self._compute_dcm_exp(step_idx, local_time)

        return dcm_out

    def _compute_dcm_ds_poly(self, step_idx, t):
        ts = self._compute_polynomial_duration(step_idx)
        time = np.clip(t, 0.0, ts)
        t_mat = np.zeros((1, 4))
        t_mat[0][0] = time**3
        t_mat[0][1] = time**2
        t_mat[0][2] = time
        t_mat[0][3] = 1.0

        return np.squeeze(np.dot(t_mat, self._dcm_P[step_idx]))

    def _compute_dcm_exp(self, step_idx, t):
        t_step = self._compute_t_step(step_idx)
        time = np.clip(t, 0.0, t_step)

        return self._vrp_list[step_idx] + np.exp((time - t_step) / self._b) * (
            self._dcm_eos_list[step_idx] - self._vrp_list[step_idx]
        )

    def _compute_step_idx(self, t):
        if t < 0.0:
            return 0

        t_ds_step_start = 0.0
        t_exp_step_end = 0.0

        for i in range(len(self._vrp_list)):
            t_ds_step_start = self._compute_ds_t_start(i)
            t_exp_step_end = self._compute_t_step_end(i) - (self._alpha_ds * self._t_ds)

            if i == 0:
                t_exp_step_end = self._compute_ds_t_end(i + 1)

            if t_ds_step_start <= t <= t_exp_step_end:
                return i

        return len(self._vrp_list) - 1

    def compute_settling_time(self):
        return -self._b * np.log(1.0 - self._percentage_settle)

    def _compute_total_trajectory_time(self):
        self._t_end = 0.0
        for i in range(len(self._vrp_list)):
            self._t_end += self._compute_t_step(i)

        self._t_end += self.compute_settling_time()

    def _compute_polynomial_matrix(
        self, ts, dcm_ini, dcm_vel_ini, dcm_end, dcm_vel_end
    ):
        mat = np.zeros((4, 4))

        mat[0, 0] = 2.0 / ts**3
        mat[0, 1] = 1.0 / ts**2
        mat[0, 2] = -2.0 / ts**3
        mat[0, 3] = 1.0 / ts**2
        mat[1, 0] = -3.0 / ts**2
        mat[1, 1] = -2.0 / ts
        mat[1, 2] = 3.0 / ts**2
        mat[1, 3] = -1.0 / ts
        mat[2, 1] = 1.0
        mat[3, 0] = 1.0

        bound = np.zeros((4, 3))
        bound[0, :] = np.copy(dcm_ini)
        bound[1, :] = np.copy(dcm_vel_ini)
        bound[2, :] = np.copy(dcm_end)
        bound[3, :] = np.copy(dcm_vel_end)

        return np.dot(mat, bound)

    def _print_boundary_conditions(self):
        for i in range(len(self._vrp_list)):
            print(
                "[{} th vrp] type: {}, pos: {}".format(
                    i, self._vrp_type_list[i], self._vrp_list[i]
                )
            )
        for i in range(len(self._dcm_ini_list)):
            print(
                "[{} th dcm] ini: {}, end: {}".format(
                    i, self._dcm_ini_list[i], self._dcm_eos_list[i]
                )
            )
        for i in range(len(self._vrp_list)):
            print(
                "[{} th ds] dcm_ini_ds: {}, dcm_end_ds: {}, dcm_vel_ini_ds: {}, dcm_vel_end_ds: {}".format(
                    i,
                    self._dcm_ini_ds_list[i],
                    self._dcm_end_ds_list[i],
                    self._dcm_vel_ini_ds_list[i],
                    self._dcm_vel_end_ds_list[i],
                )
            )

        for i in range(len(self._vrp_list)):
            print(
                "[{} th] {}, {}, {}, {}".format(
                    i,
                    self._compute_t_step_start(i),
                    self._compute_t_step_end(i),
                    self._compute_ds_t_start(i),
                    self._compute_ds_t_end(i),
                )
            )

    def _compute_t_step_start(self, step_idx):
        """
        Compute starting time of the step_idx from t_start
        """
        idx = np.clip(step_idx, 0, len(self._vrp_list) - 1)
        t_step_start = 0.0
        for i in range(idx):
            t_step_start += self._compute_t_step(i)
        return t_step_start

    def _compute_t_step_end(self, step_idx):
        idx = np.clip(step_idx, 0, len(self._vrp_list) - 1)
        return self._compute_t_step_start(step_idx) + self._compute_t_step(idx)

    def _compute_ds_t_start(self, step_idx):
        idx = np.clip(step_idx, 0, len(self._vrp_list) - 1)
        t_ds_start = self._compute_t_step_start(idx)
        if step_idx > 0:
            t_ds_start -= self._t_ds * self._alpha_ds
        return t_ds_start

    def _compute_ds_t_end(self, step_idx):
        """
        Double support ending time of the step_idx form t_start
        """
        idx = np.clip(step_idx, 0, len(self._vrp_list) - 1)
        return self._compute_ds_t_start(idx) + self._compute_polynomial_duration(idx)

    def _compute_polynomial_duration(self, step_idx):
        if step_idx == 0:
            return self._t_transfer + self._t_ds + (1 - self._alpha_ds) * self._t_ds
        elif step_idx == len(self._vrp_list) - 1:
            return self._t_ds
        else:
            return self._t_ds

    def _compute_dcm_ini_ds(self, step_idx, t_ds_ini):
        if step_idx == 0:
            return self._ini_dcm_pos
        return self._vrp_list[step_idx - 1] + np.exp(-t_ds_ini / self._b) * (
            self._dcm_ini_list[step_idx] - self._vrp_list[step_idx - 1]
        )

    def _compute_dcm_vel_ini_ds(self, step_idx, t_ds_ini):
        if step_idx == 0:
            return self._ini_dcm_vel
        return (
            (1.0 / self._b)
            * np.exp(-t_ds_ini / self._b)
            * (self._dcm_ini_list[step_idx] - self._vrp_list[step_idx - 1])
        )

    def _compute_dcm_acc_ini_ds(self, step_idx, t_ds_ini):
        if step_idx == 0:
            return np.zeros(3)
        return (
            (1.0 / self._b**2)
            * np.exp(-t_ds_ini / self._b)
            * (self._dcm_ini_list[step_idx] - self._vrp_list[step_idx - 1])
        )

    def _compute_dcm_end_ds(self, step_idx, t_ds_end):
        if step_idx == len(self._vrp_list) - 1:
            return self._vrp_list[-1]
        elif step_idx == 0:
            return self._dcm_end_ds_list[step_idx + 1]
        else:
            return self._vrp_list[step_idx] + np.exp(t_ds_end / self._b) * (
                self._dcm_ini_list[step_idx] - self._vrp_list[step_idx]
            )

    def _compute_dcm_vel_end_ds(self, step_idx, t_ds_end):
        if step_idx == len(self._vrp_list) - 1:
            return np.zeros(3)
        elif step_idx == 0:
            return self._dcm_vel_end_ds_list[step_idx + 1]
        else:
            return (
                (1.0 / self._b)
                * np.exp(t_ds_end / self._b)
                * (self._dcm_ini_list[step_idx] - self._vrp_list[step_idx])
            )

    def _compute_dcm_acc_end_ds(self, step_idx, t_ds_end):
        if step_idx == len(self._vrp_list) - 1:
            return np.zeros(3)
        elif step_idx == 0:
            return self._dcm_acc_end_ds_list[step_idx + 1]
        else:
            return (
                (1.0 / self._b**2)
                * np.exp(t_ds_end / self._b)
                * (self._dcm_ini_list[step_idx] - self._vrp_list[step_idx])
            )

    def _compute_t_step(self, step_idx):
        if self._vrp_type_list[step_idx] == VRPType.TRANSFER:
            return self._t_transfer + self._t_ds
        elif (self._vrp_type_list[step_idx] == VRPType.RF_SWING) or (
            self._vrp_type_list[step_idx] == VRPType.LF_SWING
        ):
            return self._t_ss + self._t_ds
        elif self._vrp_type_list[step_idx] == VRPType.END:
            return self._t_ds * (1 - self._alpha_ds)
        else:
            raise ValueError("vrp type is not set properly")

    def _compute_dcm_ini(self, vrp_d_i, t_step, dcm_eos_i):
        return vrp_d_i + np.exp(-t_step / self._b) * (dcm_eos_i - vrp_d_i)

    @property
    def t_transfer(self):
        return self._t_transfer

    @t_transfer.setter
    def t_transfer(self, value):
        self._t_transfer = value

    @property
    def t_ds(self):
        return self._t_ds

    @t_ds.setter
    def t_ds(self, value):
        self._t_ds = value

    @property
    def t_ss(self):
        return self._t_ss

    @t_ss.setter
    def t_ss(self, value):
        self._t_ss = value

    @property
    def percentage_settle(self):
        return self._percentage_settle

    @percentage_settle.setter
    def percentage_settle(self, value):
        self._percentage_settle = value

    @property
    def alpha_ds(self):
        return self._alpha_ds

    @alpha_ds.setter
    def alpha_ds(self, value):
        self._alpha_ds = value

    @property
    def z_vrp(self):
        return self._z_vrp

    @z_vrp.setter
    def z_vrp(self, value):
        self._z_vrp = value
        self._b = np.sqrt(self._z_vrp / 9.81)

    @property
    def robot_mass(self):
        return self._robot_mass

    @robot_mass.setter
    def robot_mass(self, value):
        self._robot_mass = value

    @property
    def t_start(self):
        return self._t_start

    @t_start.setter
    def t_start(self, value):
        self._t_start = value

    @property
    def t_end(self):
        return self._t_end
