import numpy as np

from pnc.draco_pnc.state_machine import LocomanipulationState
from pnc.interrupt_logic import InterruptLogic

COM_VEL_THRE = 0.01


class DracoManipulationInterruptLogic(InterruptLogic):
    def __init__(self, ctrl_arch):
        super(DracoManipulationInterruptLogic, self).__init__()
        self._control_architecture = ctrl_arch

        self._lh_target_pos = np.array([0.0, 0.0, 0.0])
        self._lh_waypoint_pos = np.array([0.0, 0.0, 0.0])
        self._lh_target_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._rh_target_pos = np.array([0.0, 0.0, 0.0])
        self._rh_waypoint_pos = np.array([0.0, 0.0, 0.0])
        self._rh_target_quat = np.array([0.0, 0.0, 0.0, 1.0])

        self._com_displacement_x = 0.0
        self._com_displacement_y = 0.0

        self._b_walk_in_progress = False
        self._b_walk_ready = False
        self._b_left_hand_ready = False
        self._b_right_hand_ready = False

        self._standby = False

        self._walk_forward = False
        self._walk_in_place = False
        self._walk_backward = False
        self._walk_in_x = False
        self._walk_in_y = False

        self._strafe_left = False
        self._strafe_right = False

        self._turn_left = False
        self._turn_right = False

        self._release = False

    @property
    def com_displacement_x(self):
        return self._com_displacement_x

    @com_displacement_x.setter
    def com_displacement_x(self, value):
        self._com_displacement_x = value

    @property
    def com_displacement_y(self):
        return self._com_displacement_y

    @com_displacement_y.setter
    def com_displacement_y(self, value):
        self._com_displacement_y = value

    @property
    def lh_target_pos(self):
        return self._lh_target_pos

    @lh_target_pos.setter
    def lh_target_pos(self, value):
        self._lh_target_pos = value

    @property
    def rh_target_pos(self):
        return self._rh_target_pos

    @rh_target_pos.setter
    def rh_target_pos(self, value):
        self._rh_target_pos = value

    @property
    def rh_waypoint_pos(self):
        return self._rh_waypoint_pos

    @rh_waypoint_pos.setter
    def rh_waypoint_pos(self, value):
        self._rh_waypoint_pos = value

    @property
    def lh_waypoint_pos(self):
        return self._lh_waypoint_pos

    @lh_waypoint_pos.setter
    def lh_waypoint_pos(self, value):
        self._lh_waypoint_pos = value

    @property
    def lh_target_quat(self):
        return self._lh_target_quat

    @lh_target_quat.setter
    def lh_target_quat(self, value):
        self._lh_target_quat = value

    @property
    def rh_target_quat(self):
        return self._rh_target_quat

    @rh_target_quat.setter
    def rh_target_quat(self, value):
        self._rh_target_quat = value

    @property
    def b_walk_ready(self):
        com_vel = self._control_architecture._robot.get_com_lin_vel()
        if (
            np.linalg.norm(com_vel) < COM_VEL_THRE
            and self._control_architecture.state == LocomanipulationState.BALANCE
        ):
            self._b_walk_ready = True
        else:
            self._b_walk_ready = False
        return self._b_walk_ready

    @property
    def b_left_hand_ready(self):
        com_vel = self._control_architecture._robot.get_com_lin_vel()
        if (
            np.linalg.norm(com_vel) < COM_VEL_THRE
            and self._control_architecture.state == LocomanipulationState.BALANCE
        ):
            self._b_left_hand_ready = True
        else:
            self._b_left_hand_ready = False
        return self._b_left_hand_ready

    @property
    def b_right_hand_ready(self):
        com_vel = self._control_architecture._robot.get_com_lin_vel()
        if (
            np.linalg.norm(com_vel) < COM_VEL_THRE
            and self._control_architecture.state == LocomanipulationState.BALANCE
        ):
            self._b_right_hand_ready = True
        else:
            self._b_right_hand_ready = False
        return self._b_right_hand_ready

    @property
    def b_walk_in_progress(self):
        if self._control_architecture.state_machine[
            LocomanipulationState.BALANCE
        ].walking_trigger:
            self._b_walk_in_progress = True
        else:
            self._b_walk_in_progress = False
        return self._b_walk_in_progress

    def process_interrupts(self):
        if self._standby:
            self._control_architecture._manipulation.rh_target_pos = self._rh_target_pos
            self._control_architecture._manipulation.rh_waypoint_pos = (
                self._rh_waypoint_pos
            )
            self._control_architecture._manipulation.rh_target_quat = (
                self._rh_target_quat
            )

            self._control_architecture._manipulation.lh_target_pos = self._lh_target_pos
            self._control_architecture._manipulation.lh_waypoint_pos = (
                self._lh_waypoint_pos
            )
            self._control_architecture._manipulation.lh_target_quat = (
                self._lh_target_quat
            )

            if self._control_architecture.state == LocomanipulationState.BALANCE:
                self._control_architecture.state_machine[
                    LocomanipulationState.BALANCE
                ].rhand_task_trans_trigger = True
                self._control_architecture.state_machine[
                    LocomanipulationState.BALANCE
                ].lhand_task_trans_trigger = True

                if self._walk_forward:
                    self._control_architecture.dcm_tm.walk_forward()
                    self._control_architecture.state_machine[
                        LocomanipulationState.BALANCE
                    ].walking_trigger = True
                    self._walk_forward = False

                if self._walk_in_place:
                    self._control_architecture.dcm_tm.walk_in_place()
                    self._control_architecture.state_machine[
                        LocomanipulationState.BALANCE
                    ].walking_trigger = True
                    self._walk_in_place = False

                if self._strafe_left:
                    self._control_architecture.dcm_tm.strafe_left()
                    self._control_architecture.state_machine[
                        LocomanipulationState.BALANCE
                    ].walking_trigger = True
                    self._strafe_left = False

                if self._strafe_right:
                    self._control_architecture.dcm_tm.strafe_right()
                    self._control_architecture.state_machine[
                        LocomanipulationState.BALANCE
                    ].walking_trigger = True
                    self._strafe_right = False

                if self._walk_backward:
                    self._control_architecture.dcm_tm.walk_backward()
                    self._control_architecture.state_machine[
                        LocomanipulationState.BALANCE
                    ].walking_trigger = True
                    self._walk_backward = False

                if self._turn_left:
                    self._control_architecture.dcm_tm.turn_left()
                    self._control_architecture.state_machine[
                        LocomanipulationState.BALANCE
                    ].walking_trigger = True
                    self._turn_left = False

                if self._turn_right:
                    self._control_architecture.dcm_tm.turn_right()
                    self._control_architecture.state_machine[
                        LocomanipulationState.BALANCE
                    ].walking_trigger = True
                    self._turn_right = False

                if self._walk_in_x:
                    self._control_architecture.dcm_tm.walk_in_x(
                        self._com_displacement_x
                    )
                    self._control_architecture.state_machine[
                        LocomanipulationState.BALANCE
                    ].walking_trigger = True
                    self._walk_in_x = False

                if self._walk_in_y:
                    self._control_architecture.dcm_tm.walk_in_y(
                        self._com_displacement_y
                    )
                    self._control_architecture.state_machine[
                        LocomanipulationState.BALANCE
                    ].walking_trigger = True
                    self._walk_in_y = False

                if self._release:
                    self._control_architecture.state_machine[
                        LocomanipulationState.BALANCE
                    ].rhand_task_return_trigger = True
                    self._control_architecture.state_machine[
                        LocomanipulationState.BALANCE
                    ].lhand_task_return_trigger = True
                    self._release = False

        self._reset_flags()
