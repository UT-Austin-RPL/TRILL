import numpy as np

from pnc.control_architecture import ControlArchitecture
from pnc.dcm import *
from pnc.wbc.manager import *
from .tci_container import DracoManipulationTCIContainer
from .controller import DracoManipulationController
from .state_machine import *
from .state_provider import DracoManipulationStateProvider


class DracoManipulationControlArchitecture(ControlArchitecture):
    def __init__(self, robot, config):
        super(DracoManipulationControlArchitecture, self).__init__(robot)

        self._robot = robot
        self._config = config

        walking_config = self._config['Walking']
        manipulation_config = self._config['Manipulation']

        # ======================================================================
        # Initialize TCIContainer
        # ======================================================================
        self._tci_container = DracoManipulationTCIContainer(self._robot, self._config)

        # ======================================================================
        # Initialize Controller
        # ======================================================================
        self._draco_manipulation_controller = DracoManipulationController(
            self._tci_container, self._robot, self._config)

        # ======================================================================
        # Initialize Planner
        # ======================================================================
        self._dcm_planner = DCMPlanner()

        # ======================================================================
        # Initialize Task Manager
        # ======================================================================
        self._rfoot_tm = FootTrajectoryManager(
            self._tci_container.rfoot_pos_task,
            self._tci_container.rfoot_ori_task, self._robot)
        self._rfoot_tm.swing_height = walking_config['Initial Motion']['Swing Height']

        self._lfoot_tm = FootTrajectoryManager(
            self._tci_container.lfoot_pos_task,
            self._tci_container.lfoot_ori_task, self._robot)
        self._lfoot_tm.swing_height = walking_config['Initial Motion']['Swing Height']

        self._upper_body_tm = UpperBodyTrajectoryManager(
            self._tci_container.upper_body_task, self._robot)

        self._lhand_tm = HandTrajectoryManager(
            self._tci_container.lhand_pos_task,
            self._tci_container.lhand_ori_task, self._robot, 
            target_vel_max=manipulation_config['Hand Velocity Max'],
            trajectory_mode = manipulation_config['Trajectory Mode'])

        self._rhand_tm = HandTrajectoryManager(
            self._tci_container.rhand_pos_task,
            self._tci_container.rhand_ori_task, self._robot, 
            target_vel_max=manipulation_config['Hand Velocity Max'],
            trajectory_mode = manipulation_config['Trajectory Mode'])

        self._floating_base_tm = FloatingBaseTrajectoryManager(
            self._tci_container.com_task, self._tci_container.torso_ori_task,
            self._robot)

        self._dcm_tm = DCMTrajectoryManager(self._dcm_planner,
                                            self._tci_container.com_task,
                                            self._tci_container.torso_ori_task,
                                            self._robot, "l_foot_contact",
                                            "r_foot_contact")
        self._dcm_tm.nominal_com_height = walking_config['Initial Motion']['COM Height']
        self._dcm_tm.t_additional_init_transfer = walking_config['Duration']['Additional Inititial Transfer']
        self._dcm_tm.t_contact_transition = walking_config['Duration']['Contact Trans']
        self._dcm_tm.t_swing = walking_config['Duration']['Swing']
        self._dcm_tm.percentage_settle = walking_config['Percentage Settle']
        self._dcm_tm.alpha_ds = walking_config['Alpha DS']
        self._dcm_tm.nominal_footwidth = walking_config['Initial Motion']['Footwidth']
        self._dcm_tm.nominal_forward_step = walking_config['Initial Motion']['Forward Step']
        self._dcm_tm.nominal_backward_step = walking_config['Initial Motion']['Backward Step']
        self._dcm_tm.nominal_turn_radians = walking_config['Initial Motion']['Trun Angle']
        self._dcm_tm.nominal_strafe_distance = walking_config['Initial Motion']['Strafe Distance']

        self._trajectory_managers = {
            "rfoot": self._rfoot_tm,
            "lfoot": self._lfoot_tm,
            "upper_body": self._upper_body_tm,
            "lhand": self._lhand_tm,
            "rhand": self._rhand_tm,
            "floating_base": self._floating_base_tm,
            "dcm": self._dcm_tm
        }

        hiearchy_config = self._config['Whole-Body Contol']['Hierarchy']

        # ======================================================================
        # Initialize Hierarchy Manager
        # ======================================================================
        self._rfoot_pos_hm = TaskHierarchyManager(
            self._tci_container.rfoot_pos_task,
            hiearchy_config['Contact Foot'],
            hiearchy_config['Swing Foot'])

        self._lfoot_pos_hm = TaskHierarchyManager(
            self._tci_container.lfoot_pos_task,
            hiearchy_config['Contact Foot'],
            hiearchy_config['Swing Foot'])

        self._rfoot_ori_hm = TaskHierarchyManager(
            self._tci_container.rfoot_ori_task, 
            hiearchy_config['Contact Foot'],
            hiearchy_config['Swing Foot'])

        self._lfoot_ori_hm = TaskHierarchyManager(
            self._tci_container.lfoot_ori_task,
            hiearchy_config['Contact Foot'],
            hiearchy_config['Swing Foot'])

        self._lhand_pos_hm = TaskHierarchyManager(
            self._tci_container.lhand_pos_task,
            hiearchy_config['Hand Pos Max'],
            hiearchy_config['Hand Pos Min'])

        self._lhand_ori_hm = TaskHierarchyManager(
            self._tci_container.lhand_ori_task,
            hiearchy_config['Hand Quat Max'],
            hiearchy_config['Hand Quat Min'])

        self._rhand_pos_hm = TaskHierarchyManager(
            self._tci_container.rhand_pos_task,
            hiearchy_config['Hand Pos Max'],
            hiearchy_config['Hand Pos Min'])

        self._rhand_ori_hm = TaskHierarchyManager(
            self._tci_container.rhand_ori_task,
            hiearchy_config['Hand Quat Max'],
            hiearchy_config['Hand Quat Min'])

        self._hierarchy_managers = {
            "rfoot_pos": self._rfoot_pos_hm,
            "lfoot_pos": self._lfoot_pos_hm,
            "rfoot_ori": self._rfoot_ori_hm,
            "lfoot_ori": self._lfoot_ori_hm,
            "lhand_pos": self._lhand_pos_hm,
            "lhand_ori": self._lhand_ori_hm,
            "rhand_pos": self._rhand_pos_hm,
            "rhand_ori": self._rhand_ori_hm
        }

        # ======================================================================
        # Initialize Reaction Force Manager
        # ======================================================================
        self._rfoot_fm = ReactionForceManager(
            self._tci_container.rfoot_contact, config['Whole-Body Contol']['RF Z Max'])

        self._lfoot_fm = ReactionForceManager(
            self._tci_container.lfoot_contact, config['Whole-Body Contol']['RF Z Max'])

        self._reaction_force_managers = {
            "rfoot": self._rfoot_fm,
            "lfoot": self._lfoot_fm
        }

        # ======================================================================
        # Initialize State Machines
        # ======================================================================
        self._state_machine[LocomanipulationState.STAND] = DoubleSupportStand(
            LocomanipulationState.STAND, self._trajectory_managers,
            self._hierarchy_managers, self._reaction_force_managers, self._robot)
        self._state_machine[LocomanipulationState.STAND].end_time = walking_config['Duration']['Initial Stand']
        self._state_machine[LocomanipulationState.STAND].rf_z_max_time = walking_config['Duration']['RF Z Max']
        self._state_machine[LocomanipulationState.STAND].com_height_des = walking_config['Initial Motion']['COM Height']

        self._state_machine[
            LocomanipulationState.BALANCE] = DoubleSupportBalance(
                LocomanipulationState.BALANCE, self._trajectory_managers,
                self._hierarchy_managers, self._reaction_force_managers, self._robot)

        self._state_machine[LocomanipulationState.
                            LF_CONTACT_TRANS_START] = ContactTransitionStart(
                                LocomanipulationState.LF_CONTACT_TRANS_START,
                                self._trajectory_managers,
                                self._hierarchy_managers,
                                self._reaction_force_managers,
                                Footstep.LEFT_SIDE, self._robot)

        self._state_machine[
            LocomanipulationState.LF_CONTACT_TRANS_END] = ContactTransitionEnd(
                LocomanipulationState.LF_CONTACT_TRANS_END,
                self._trajectory_managers, self._hierarchy_managers,
                self._reaction_force_managers, Footstep.LEFT_SIDE, self._robot)

        self._state_machine[
            LocomanipulationState.LF_SWING] = SingleSupportSwing(
                LocomanipulationState.LF_SWING, self._trajectory_managers,
                Footstep.LEFT_SIDE, self._robot)

        self._state_machine[LocomanipulationState.
                            RF_CONTACT_TRANS_START] = ContactTransitionStart(
                                LocomanipulationState.RF_CONTACT_TRANS_START,
                                self._trajectory_managers,
                                self._hierarchy_managers,
                                self._reaction_force_managers,
                                Footstep.RIGHT_SIDE, self._robot)

        self._state_machine[
            LocomanipulationState.RF_CONTACT_TRANS_END] = ContactTransitionEnd(
                LocomanipulationState.RF_CONTACT_TRANS_END,
                self._trajectory_managers, self._hierarchy_managers,
                self._reaction_force_managers, Footstep.RIGHT_SIDE,
                self._robot)

        self._state_machine[
            LocomanipulationState.RF_SWING] = SingleSupportSwing(
                LocomanipulationState.RF_SWING, self._trajectory_managers,
                Footstep.RIGHT_SIDE, self._robot)

        self._manipulation = Manipulation(LocomanipulationState.DH_MANIPULATION, self._trajectory_managers,
                                          self._hierarchy_managers, self._reaction_force_managers,
                                          self._robot)
        self._manipulation.moving_duration = manipulation_config['Duration']['Reaching']
        self._manipulation.rh_target_pos = np.array(manipulation_config['Initial Target']['RH Pos'])
        self._manipulation.rh_target_quat = np.array(manipulation_config['Initial Target']['RH Quat'])
        self._manipulation.lh_target_pos = np.array(manipulation_config['Initial Target']['LH Pos'])
        self._manipulation.lh_target_quat = np.array(manipulation_config['Initial Target']['LH Quat'])
        self._manipulation.trans_duration = manipulation_config['Duration']['Reaching Trans']

        # Set Starting State
        self._state = LocomanipulationState.STAND
        self._prev_state = LocomanipulationState.STAND
        self._b_state_first_visit = True
        self._b_manipulation_first_visit = True

        self._sp = DracoManipulationStateProvider()

    def get_command(self):
        if self._b_state_first_visit:
            self._state_machine[self._state].first_visit()
            self._b_state_first_visit = False

        # Update State Machine
        self._state_machine[self._state].one_step()

        if self._b_manipulation_first_visit:
            self._manipulation.first_visit()
            self._b_manipulation_first_visit = False

        # Update State Machine
        self._manipulation.one_step()

        # Update State Machine Independent Trajectories
        self._upper_body_tm.use_nominal_upper_body_joint_pos(
            self._sp.nominal_joint_pos)

        # Get Whole Body Control Commands
        command = self._draco_manipulation_controller.get_command()

        if self._state_machine[self._state].end_of_state():
            self._state_machine[self._state].last_visit()
            self._prev_state = self._state
            self._state = self._state_machine[self._state].get_next_state()
            self._b_state_first_visit = True

        if self._manipulation.end_of_state():
            self._manipulation.last_visit()
            self._b_manipulation_first_visit = True

        return command

    @property
    def dcm_tm(self):
        return self._dcm_tm

    @property
    def state_machine(self):
        return self._state_machine
