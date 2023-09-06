import numpy as np

from pnc.data_saver import DataSaver
from pnc.wbc.ihwbc.ihwbc import IHWBC
# from pnc.wbc.ihwbc.ihwbc2 import IHWBC2
from pnc.wbc.ihwbc.joint_integrator import JointIntegrator


class DracoManipulationController(object):
    def __init__(self, tci_container, robot, config):
        self._tci_container = tci_container
        self._robot = robot
        self._config = config

        # Initialize WBC
        l_jp_idx, l_jd_idx, r_jp_idx, r_jd_idx, neck_pitch_idx = self._robot.get_q_dot_idx(
            ['l_knee_fe_jp', 'l_knee_fe_jd', 'r_knee_fe_jp', 'r_knee_fe_jd', 'neck_pitch'])
        act_list = [False] * robot.n_floating + [True] * robot.n_a
        act_list[l_jp_idx] = False
        act_list[r_jp_idx] = False
        act_list[neck_pitch_idx] = False

        n_q_dot = len(act_list)
        n_active = np.count_nonzero(np.array(act_list))
        n_passive = n_q_dot - n_active - 6

        self._sa = np.zeros((n_active, n_q_dot))
        self._sv = np.zeros((n_passive, n_q_dot))

        self._sd = np.zeros((n_passive, n_q_dot))
        j, k = 0, 0
        for i in range(n_q_dot):
            if i >= 6:
                if act_list[i]:
                    self._sa[j, i] = 1.
                    j += 1
                else:
                    self._sv[k, i] = 1.
                    k += 1

        self._sd[0, l_jd_idx] = 1.
        self._sd[1, r_jd_idx] = 1.

        self._sf = np.zeros((6, n_q_dot))
        self._sf[0:6, 0:6] = np.eye(6)

        self._ihwbc = IHWBC(self._sf, self._sa, self._sv, self._config['Simulation']['Save Data'])

        ###consider transmission constraint
        # self._ihwbc = IHWBC2(self._sf, self._sa, self._sv, self._sd,
        # PnCConfig.SAVE_DATA)

        if self._config['Whole-Body Contol']['Use Torque Limit']:
            self._ihwbc.trq_limit = np.dot(self._sa[:, 6:],
                                           self._robot.joint_trq_limit)
        self._ihwbc.lambda_q_ddot = self._config['Whole-Body Contol']['Reguralization']['Q ddot']
        self._ihwbc.lambda_rf = self._config['Whole-Body Contol']['Reguralization']['RF']

        # Initialize Joint Integrator
        self._joint_integrator = JointIntegrator(robot.n_a,
                                                 self._config['Simulation']['Control Period'])
        self._joint_integrator.pos_cutoff_freq = self._config['Whole-Body Contol']['Integration']['Cutoff Frequency']['Joint Pos']
        self._joint_integrator.vel_cutoff_freq = self._config['Whole-Body Contol']['Integration']['Cutoff Frequency']['Joint Vel']
        self._joint_integrator.max_pos_err = self._config['Whole-Body Contol']['Integration']['Joint Pos Error Max']
        self._joint_integrator.joint_pos_limit = self._robot.joint_pos_limit
        self._joint_integrator.joint_vel_limit = self._robot.joint_vel_limit

        self._b_first_visit = True

        if self._config['Simulation']['Save Data']:
            self._data_saver = DataSaver()

    def get_command(self):
        if self._b_first_visit:
            self.first_visit()

        # Dynamics properties
        mass_matrix = self._robot.get_mass_matrix()
        mass_matrix_inv = np.linalg.inv(mass_matrix)
        coriolis = self._robot.get_coriolis()
        gravity = self._robot.get_gravity()
        self._ihwbc.update_setting(mass_matrix, mass_matrix_inv, coriolis,
                                   gravity)
        # Task, Contact, and Internal Constraint Setup
        w_hierarchy_list = []
        for task in self._tci_container.task_list:
            task.update_jacobian()
            task.update_cmd()
            w_hierarchy_list.append(task.w_hierarchy)
        self._ihwbc.w_hierarchy = np.array(w_hierarchy_list)
        for contact in self._tci_container.contact_list:
            contact.update_contact()
        for internal_constraint in self._tci_container.internal_constraint_list:
            internal_constraint.update_internal_constraint()
        # WBC commands
        joint_trq_cmd, joint_acc_cmd, rf_cmd = self._ihwbc.solve(
            self._tci_container.task_list, self._tci_container.contact_list,
            self._tci_container.internal_constraint_list, None,
            self._config['Simulation']['Verbose'])
        ###consider transmission constraint
        # joint_trq_cmd, joint_acc_cmd, rf_cmd = self._ihwbc.solve(
        # self._tci_container.task_list, self._tci_container.contact_list,
        # self._tci_container.internal_constraint_list, None, True, True)
        joint_trq_cmd = np.dot(self._sa[:, 6:].transpose(), joint_trq_cmd)
        joint_acc_cmd = np.dot(self._sa[:, 6:].transpose(), joint_acc_cmd)
        # Double integration
        joint_vel_cmd, joint_pos_cmd = self._joint_integrator.integrate(
            joint_acc_cmd, self._robot.joint_velocities,
            self._robot.joint_positions)

        if self._config['Simulation']['Save Data']:
            self._data_saver.add('joint_trq_cmd', joint_trq_cmd)

        command = self._robot.create_cmd_ordered_dict(joint_pos_cmd,
                                                      joint_vel_cmd,
                                                      joint_trq_cmd)
        
        command['reaction_force'] = rf_cmd

        return command

    def first_visit(self):
        joint_pos_ini = self._robot.joint_positions
        self._joint_integrator.initialize_states(np.zeros(self._robot.n_a),
                                                 joint_pos_ini)

        self._b_first_visit = False
