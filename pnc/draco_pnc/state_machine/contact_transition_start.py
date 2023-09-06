from scipy.spatial.transform import Rotation as R

from pnc.draco_pnc.state_machine import LocomanipulationState
from pnc.dcm import Footstep
from pnc.state_machine import StateMachine
from pnc.wbc.manager.dcm_trajectory_manager import DCMTransferType
from pnc.draco_pnc.state_provider import DracoManipulationStateProvider


class ContactTransitionStart(StateMachine):
    def __init__(self, id, tm, hm, fm, leg_side, robot):
        super(ContactTransitionStart, self).__init__(id, robot)
        self._trajectory_managers = tm
        self._hierarchy_managers = hm
        self._force_managers = fm
        self._leg_side = leg_side
        self._sp = DracoManipulationStateProvider()
        self._start_time = 0.
        self._planning_id = 0

    def first_visit(self):
        if self._leg_side == Footstep.RIGHT_SIDE:
            print("[LocomanipulationState] RightLeg ContactTransitionStart")
        else:
            print("[LocomanipulationState] LeftLeg ContactTransitionStart")
        self._start_time = self._sp.curr_time

        # Initialize Reaction Force Ramp to Max
        for fm in self._force_managers.values():
            fm.initialize_ramp_to_max(
                self._sp.curr_time,
                self._trajectory_managers["dcm"].compute_rf_z_ramp_up_time())

        # Initialize Hierarchy Ramp to Max
        self._hierarchy_managers["rfoot_pos"].initialize_ramp_to_max(
            self._sp.curr_time,
            self._trajectory_managers["dcm"].compute_rf_z_ramp_up_time())
        self._hierarchy_managers["lfoot_pos"].initialize_ramp_to_max(
            self._sp.curr_time,
            self._trajectory_managers["dcm"].compute_rf_z_ramp_up_time())
        self._hierarchy_managers["rfoot_ori"].initialize_ramp_to_max(
            self._sp.curr_time,
            self._trajectory_managers["dcm"].compute_rf_z_ramp_up_time())
        self._hierarchy_managers["lfoot_ori"].initialize_ramp_to_max(
            self._sp.curr_time,
            self._trajectory_managers["dcm"].compute_rf_z_ramp_up_time())

        # Check if it is the last footstep
        if self._trajectory_managers["dcm"].no_reaming_steps():
            self._end_time = self._trajectory_managers[
                "dcm"].compute_final_contact_transfer_time()
        else:
            transfer_type = DCMTransferType.MID
            self._end_time = self._trajectory_managers[
                "dcm"].compute_rf_z_ramp_up_time()

            if self._sp.prev_state == LocomanipulationState.BALANCE:
                transfer_type = DCMTransferType.INI
                self._end_time = self._trajectory_managers[
                    "dcm"].compute_ini_contact_transfer_time(
                    ) - self._trajectory_managers[
                        "dcm"].compute_rf_z_ramp_down_time()

                # TODO: Replanning
                torso_quat = R.from_matrix(
                    self._robot.get_link_iso("torso_link")[0:3,
                                                           0:3]).as_quat()
                self._trajectory_managers["dcm"].initialize(
                    self._sp.curr_time, transfer_type, torso_quat,
                    self._sp.dcm, self._sp.dcm_vel)
                self._trajectory_managers["dcm"].save_trajectory(
                    self._planning_id)
                self._planning_id += 1

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update max normal reaction forces
        for fm in self._force_managers.values():
            fm.update_ramp_to_max(self._sp.curr_time)

        # Update task hieararchy weights
        self._hierarchy_managers["rfoot_pos"].update_ramp_to_max(
            self._sp.curr_time)
        self._hierarchy_managers["lfoot_pos"].update_ramp_to_max(
            self._sp.curr_time)
        self._hierarchy_managers["rfoot_ori"].update_ramp_to_max(
            self._sp.curr_time)
        self._hierarchy_managers["lfoot_ori"].update_ramp_to_max(
            self._sp.curr_time)

        # Update floating base task
        self._trajectory_managers["dcm"].update_floating_base_task_desired(
            self._sp.curr_time)

        # Update foot task
        self._trajectory_managers["lfoot"].use_current()
        self._trajectory_managers["rfoot"].use_current()

        # self._trajectory_managers["lhand"].use_current()
        # self._trajectory_managers["rhand"].use_current()

    def last_visit(self):
        pass

    def end_of_state(self):
        if self._state_machine_time >= self._end_time:
            return True
        else:
            return False

    def get_next_state(self):
        if self._trajectory_managers["dcm"].no_reaming_steps():
            return LocomanipulationState.BALANCE
        else:
            if self._leg_side == Footstep.LEFT_SIDE:
                return LocomanipulationState.LF_CONTACT_TRANS_END
            elif self._leg_side == Footstep.RIGHT_SIDE:
                return LocomanipulationState.RF_CONTACT_TRANS_END
