import numpy as np

from pnc.state_machine import StateMachine
from pnc.draco_pnc.state_provider import DracoManipulationStateProvider
from pnc.draco_pnc.state_machine import LocomanipulationState
from util import geom


class Manipulation(StateMachine):
    def __init__(self, id, tm, hm, fm, robot):
        super().__init__(id, robot)
        self._trajectory_managers = tm
        self._hierarchy_managers = hm
        self._sp = DracoManipulationStateProvider(robot)
        self._start_time = 0.
        self._moving_duration = 0.0
        self._trans_duration = 0.
        self._rh_target_pos = np.zeros(3)
        self._rh_target_quat = np.zeros(4)
        self._lh_target_pos = np.zeros(3)
        self._lh_target_quat = np.zeros(4)

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update Hierarchy
        if self._state_id == LocomanipulationState.BALANCE:
            self._hierarchy_managers["lhand_pos"].update_ramp_to_max(
                self._sp.curr_time)
            self._hierarchy_managers["lhand_ori"].update_ramp_to_max(
                self._sp.curr_time)
            self._hierarchy_managers["rhand_pos"].update_ramp_to_max(
                self._sp.curr_time)
            self._hierarchy_managers["rhand_ori"].update_ramp_to_max(
                self._sp.curr_time)
        else:
            self._hierarchy_managers["lhand_pos"].update_ramp_to_min(
                self._sp.curr_time)
            self._hierarchy_managers["lhand_ori"].update_ramp_to_min(
                self._sp.curr_time)
            self._hierarchy_managers["rhand_pos"].update_ramp_to_min(
                self._sp.curr_time)
            self._hierarchy_managers["rhand_ori"].update_ramp_to_min(
                self._sp.curr_time)

        # Update Hand Task
        self._trajectory_managers['lhand'].update_hand_pose(
        self._sp.curr_time)
        self._trajectory_managers['rhand'].update_hand_pose(
        self._sp.curr_time)


    def first_visit(self):
        self._start_time = self._sp.curr_time

        target_rh_iso = np.eye(4)
        target_rh_iso[0:3, 0:3] = geom.quat_to_rot(self._rh_target_quat)
        target_rh_iso[0:3, 3] = self._rh_target_pos

        target_lh_iso = np.eye(4)
        target_lh_iso[0:3, 0:3] = geom.quat_to_rot(self._lh_target_quat)
        target_lh_iso[0:3, 3] = self._lh_target_pos

        self._trajectory_managers[
            'rhand'].initialize_hand_pose(
                self._start_time, self._moving_duration,
                target_rh_iso)

        self._trajectory_managers[
            'lhand'].initialize_hand_pose(
                self._start_time, self._moving_duration,
                target_lh_iso)

        if self._state_id == LocomanipulationState.BALANCE:
            self._hierarchy_managers["rhand_pos"].initialize_ramp_to_max(
                self._sp.curr_time, self._trans_duration)
            self._hierarchy_managers["rhand_ori"].initialize_ramp_to_max(
                self._sp.curr_time, self._trans_duration)

            self._hierarchy_managers["lhand_pos"].initialize_ramp_to_max(
                self._sp.curr_time, self._trans_duration)
            self._hierarchy_managers["lhand_ori"].initialize_ramp_to_max(
                self._sp.curr_time, self._trans_duration)
        else:
            self._hierarchy_managers["lhand_pos"].initialize_ramp_to_min(
                self._sp.curr_time, self._trans_duration)
            self._hierarchy_managers["lhand_ori"].initialize_ramp_to_min(
                self._sp.curr_time, self._trans_duration)
            self._hierarchy_managers["rhand_pos"].initialize_ramp_to_min(
                self._sp.curr_time, self._trans_duration)
            self._hierarchy_managers["rhand_ori"].initialize_ramp_to_min(
                self._sp.curr_time, self._trans_duration)


    def last_visit(self):
        pass

    def end_of_state(self):
        if self._state_machine_time > self._moving_duration + 0.1:
            return True
        else:
            return False

    def get_next_state(self):
        return LocomanipulationState.BALANCE

    @property
    def moving_duration(self):
        return self._moving_duration

    @moving_duration.setter
    def moving_duration(self, value):
        self._moving_duration = value

    @property
    def trans_duration(self):
        return self._trans_duration

    @trans_duration.setter
    def trans_duration(self, value):
        self._trans_duration = value

    @property
    def rh_target_pos(self):
        return self._rh_target_pos

    @rh_target_pos.setter
    def rh_target_pos(self, value):
        self._rh_target_pos = value

    @property
    def rh_target_quat(self):
        return self._rh_target_quat

    @rh_target_quat.setter
    def rh_target_quat(self, value):
        self._rh_target_quat = value

    @property
    def lh_target_pos(self):
        return self._lh_target_pos

    @lh_target_pos.setter
    def lh_target_pos(self, value):
        self._lh_target_pos = value

    @property
    def lh_target_quat(self):
        return self._lh_target_quat

    @lh_target_quat.setter
    def lh_target_quat(self, value):
        self._lh_target_quat = value

