from pnc.dcm import Footstep
from pnc.draco_pnc.state_machine import LocomanipulationState
from pnc.draco_pnc.state_provider import DracoManipulationStateProvider
from pnc.state_machine import StateMachine


class SingleSupportSwing(StateMachine):
    def __init__(self, id, tm, leg_side, robot):
        super(SingleSupportSwing, self).__init__(id, robot)
        self._trajectory_managers = tm
        self._leg_side = leg_side
        self._sp = DracoManipulationStateProvider(robot)
        self._start_time = 0.0

    def first_visit(self):
        if self._leg_side == Footstep.RIGHT_SIDE:
            print("[LocomanipulationState] RightLeg SingleSupportSwing")
        else:
            print("[LocomanipulationState] LeftLeg SingleSupportSwing")
        self._start_time = self._sp.curr_time
        self._end_time = self._trajectory_managers["dcm"].compute_swing_time()

        footstep_idx = self._trajectory_managers["dcm"].curr_footstep_idx

        if self._leg_side == Footstep.RIGHT_SIDE:
            self._trajectory_managers["rfoot"].initialize_swing_foot_trajectory(
                self._sp.curr_time,
                self._end_time,
                self._trajectory_managers["dcm"].footstep_list[footstep_idx],
            )
        else:
            self._trajectory_managers["lfoot"].initialize_swing_foot_trajectory(
                self._sp.curr_time,
                self._end_time,
                self._trajectory_managers["dcm"].footstep_list[footstep_idx],
            )

    def one_step(self):
        self._state_machine_time = self._sp.curr_time - self._start_time

        # Update floating base task
        self._trajectory_managers["dcm"].update_floating_base_task_desired(
            self._sp.curr_time
        )

        # Update foot task
        if self._leg_side == Footstep.LEFT_SIDE:
            self._trajectory_managers["lfoot"].update_swing_foot_desired(
                self._sp.curr_time
            )
            self._trajectory_managers["rfoot"].use_current()
        else:
            self._trajectory_managers["lfoot"].use_current()
            self._trajectory_managers["rfoot"].update_swing_foot_desired(
                self._sp.curr_time
            )

    def last_visit(self):
        self._trajectory_managers["dcm"].increment_step_idx()

    def end_of_state(self):
        """
        if self._state_machine_time >= self._end_time:
            return True
        else:
            if self._state_machine_time >= 0.5 * self._end_time:
                if self._leg_side == Footstep.LEFT_SIDE:
                    if self._sp.b_lf_contact:
                        print("Early left foot contact at {}/{}".format(
                            self._state_machine_time, self._end_time))
                        return True
                else:
                    if self._sp.b_rf_contact:
                        print("Early right foot contact at {}/{}".format(
                            self._state_machine_time, self._end_time))
                        return True
            return False
        """
        if self._state_machine_time >= self._end_time:
            return True
        else:
            return False

    def get_next_state(self):
        b_next, next_side = self._trajectory_managers["dcm"].next_step_side()
        if b_next:
            if next_side == Footstep.LEFT_SIDE:
                return LocomanipulationState.LF_CONTACT_TRANS_START
            else:
                return LocomanipulationState.RF_CONTACT_TRANS_START
        else:
            if self._leg_side == Footstep.LEFT_SIDE:
                return LocomanipulationState.LF_CONTACT_TRANS_START
            else:
                return LocomanipulationState.RF_CONTACT_TRANS_START
