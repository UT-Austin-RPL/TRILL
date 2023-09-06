from pnc.draco_pnc.state_machine import LocomanipulationState
from pnc.state_machine import StateMachine
from pnc.dcm import Footstep
from pnc.draco_pnc.state_provider import DracoManipulationStateProvider


class ContactTransitionEnd(StateMachine):
    def __init__(self, id, tm, hm, fm, leg_side, robot):
        super(ContactTransitionEnd, self).__init__(id, robot)
        self._trajectory_managers = tm
        self._hierarchy_managers = hm
        self._force_managers = fm
        self._leg_side = leg_side
        self._sp = DracoManipulationStateProvider()
        self._start_time = 0.

    def first_visit(self):
        if self._leg_side == Footstep.RIGHT_SIDE:
            print("[LocomanipulationState] RightLeg ContactTransitionEnd")
        else:
            print("[LocomanipulationState] LeftLeg ContactTransitionEnd")
        self._start_time = self._sp.curr_time
        self._end_time = self._trajectory_managers[
            "dcm"].compute_rf_z_ramp_down_time()

        if self._leg_side == Footstep.LEFT_SIDE:
            self._force_managers["lfoot"].initialize_ramp_to_min(
                self._sp.curr_time, self._end_time)
            self._hierarchy_managers["lfoot_pos"].initialize_ramp_to_min(
                self._sp.curr_time, self._end_time)
            self._hierarchy_managers["lfoot_ori"].initialize_ramp_to_min(
                self._sp.curr_time, self._end_time)
        elif self._leg_side == Footstep.RIGHT_SIDE:
            self._force_managers["rfoot"].initialize_ramp_to_min(
                self._sp.curr_time, self._end_time)
            self._hierarchy_managers["rfoot_pos"].initialize_ramp_to_min(
                self._sp.curr_time, self._end_time)
            self._hierarchy_managers["rfoot_ori"].initialize_ramp_to_min(
                self._sp.curr_time, self._end_time)
        else:
            raise ValueError("Wrong Leg Side")

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update max normal reaction forces and task hieararchy weights
        if self._leg_side == Footstep.LEFT_SIDE:
            self._force_managers["lfoot"].update_ramp_to_min(
                self._sp.curr_time)
            self._hierarchy_managers["lfoot_pos"].update_ramp_to_min(
                self._sp.curr_time)
            self._hierarchy_managers["lfoot_ori"].update_ramp_to_min(
                self._sp.curr_time)
        elif self._leg_side == Footstep.RIGHT_SIDE:
            self._force_managers["rfoot"].update_ramp_to_min(
                self._sp.curr_time)
            self._hierarchy_managers["rfoot_pos"].update_ramp_to_min(
                self._sp.curr_time)
            self._hierarchy_managers["rfoot_ori"].update_ramp_to_min(
                self._sp.curr_time)

        else:
            raise ValueError("Wrong Leg Side")

        # Update floating base task
        self._trajectory_managers["dcm"].update_floating_base_task_desired(
            self._sp.curr_time)

        # Update foot task
        self._trajectory_managers["lfoot"].use_current()
        self._trajectory_managers["rfoot"].use_current()

    def last_visit(self):
        pass

    def end_of_state(self):
        if self._state_machine_time >= self._end_time:
            return True
        else:
            return False

    def get_next_state(self):
        if self._leg_side == Footstep.LEFT_SIDE:
            return LocomanipulationState.LF_SWING
        else:
            return LocomanipulationState.RF_SWING
