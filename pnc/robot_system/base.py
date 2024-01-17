import abc
from collections import OrderedDict

import numpy as np


class RobotSystem(abc.ABC):
    def __init__(self, urdf_file, package_name, b_fixed_base, b_print_robot_info=False):
        """
        Base RobotSystem Class

        Parameters
        ----------
        urdf_file (str):
            urdf file
        package_name (str):
            mesh directory
        """
        self._b_fixed_base = b_fixed_base

        self._n_floating = 0
        self._n_q = 0
        self._n_q_dot = 0
        self._n_a = 0

        self._total_mass = 0.0

        self._joint_pos_limit = None
        self._joint_vel_limit = None
        self._joint_trq_limit = None

        self._joint_id = OrderedDict()
        self._link_id = OrderedDict()

        self._config_robot(urdf_file, package_name)

        if b_print_robot_info:
            print("=" * 80)
            print("PnCRobot")
            print(
                "nq: ",
                self._n_q,
                ", nv: ",
                self._n_q_dot,
                ", na: ",
                self._n_a,
                ", nvirtual: ",
                self._n_floating,
            )
            print("+" * 80)
            print("Joint Infos")
            for key in self._joint_id.keys():
                print("\t", key)
            print("+" * 80)
            print("Link Infos")
            for key in self._link_id.keys():
                print("\t", key)
            print("=" * 80)

        self._joint_positions = None
        self._joint_velocities = None

        self._Ig = np.zeros((6, 6))
        self._Ag = np.zeros((6, self._n_q_dot))
        self._hg = np.zeros(6)

    @property
    def n_floating(self):
        return self._n_floating

    @property
    def n_q(self):
        return self._n_q

    @property
    def n_q_dot(self):
        return self._n_q_dot

    @property
    def n_a(self):
        return self._n_a

    @property
    def total_mass(self):
        return self._total_mass

    @property
    def joint_pos_limit(self):
        return self._joint_pos_limit

    @property
    def joint_vel_limit(self):
        return self._joint_vel_limit

    @property
    def joint_trq_limit(self):
        return self._joint_trq_limit

    @property
    def joint_id(self):
        return self._joint_id

    @property
    def link_id(self):
        return self._link_id

    @property
    def Ig(self):
        return self._Ig

    @property
    def Ag(self):
        return self._Ag

    @property
    def hg(self):
        return self._hg

    @property
    def joint_positions(self):
        return self._joint_positions

    @property
    def joint_velocities(self):
        return self._joint_velocities

    @abc.abstractmethod
    def _update_centroidal_quantities(self):
        """
        Update Ig, Ag, hg:
            hg = Ig * centroid_velocity = Ag * qdot
        Note that all quantities are represented at the world frame
        """
        pass

    @abc.abstractmethod
    def _config_robot(self, urdf_file, package_dir):
        """
        Configure following properties:
            n_floating (int):
                Number of floating joints
            n_q (int):
                Size of joint positions in generalized coordinate
            n_q_dot (int):
                Size of joint velocities in generalized coordinate
            n_a (int):
                Size of actuation in generalized coordinate
            total_mass (double):
                Total mass of the robot
            joint_pos_limit (np.ndarray):
                Joint position limits. Size of (n_a, 2)
            joint_vel_limit (np.ndarray):
                Joint velocity limits. Size of (n_a, 2)
            joint_trq_limit (np.ndarray):
                Joint torque limits. Size of (n_a, 2)
            joint_id (OrderedDict):
                Key: joint name, Value: joint indicator
            floating_id (OrderedDict):
                Key: floating joint name, Value: joint indicator
            link_id (OrderedDict):
                Key: link name, Value: link indicator

        Parameters
        ----------
        urdf_file (str): urdf path
        """
        pass

    @abc.abstractmethod
    def get_q_idx(self, joint_id):
        """
        Get joint index in generalized coordinate

        Parameters
        ----------
        joint_id (str or list of str)

        Returns
        -------
        joint_idx (int or list of int)
        """
        pass

    @abc.abstractmethod
    def get_q_dot_idx(self, joint_id):
        """
        Get joint velocity index in generalized coordinate

        Parameters
        ----------
        joint_id (str or list of str)

        Returns
        -------
        joint_idx (int or list of int)
        """
        pass

    @abc.abstractmethod
    def get_joint_idx(self, joint_id):
        """
        Get joint index in generalized coordinate

        Parameters
        ----------
        joint_id (str or list of str)

        Returns
        -------
        joint_idx (int or list of int)
        """
        pass

    @abc.abstractmethod
    def create_cmd_ordered_dict(self, joint_pos_cmd, joint_vel_cmd, joint_trq_cmd):
        """
        Create command ordered dict

        Parameters
        ----------
        joint_pos_cmd (np.array):
            Joint Pos Cmd
        joint_vel_cmd (np.array):
            Joint Vel Cmd
        joint_trq_cmd (np.array):
            Joint Trq Cmd

        Returns
        -------
        command (OrderedDict)
        """
        pass

    @abc.abstractmethod
    def update_system(
        self,
        base_com_pos,
        base_com_quat,
        base_com_lin_vel,
        base_com_ang_vel,
        base_joint_pos,
        base_joint_quat,
        base_joint_lin_vel,
        base_joint_ang_vel,
        joint_pos,
        joint_vel,
        b_cent=False,
    ):
        """
        Update generalized coordinate

        Parameters
        ----------
        base_pos (np.array): Root pos, None if the robot is fixed in the world
        base_quat (np.array): Root quat
        base_lin_vel (np.array): Root linear velocity
        base_ang_vel (np.array): Root angular velocity
        joint_pos (OrderedDict): Actuator pos
        joint_vel (OrderedDict): Actuator vel
        b_cent (Bool): Whether updating centroidal frame or not
        """
        pass

    @abc.abstractmethod
    def get_q(self):
        """
        Returns
        -------
        q (np.array): positions in generalized coordinate
        """
        pass

    @abc.abstractmethod
    def get_q_dot(self):
        """
        Returns
        -------
        qdot (np.array): velocities in generalized coordinate
        """
        pass

    @abc.abstractmethod
    def get_mass_matrix(self):
        """
        Returns
        -------
        A (np.array): Mass matrix in generalized coordinate
        """
        pass

    @abc.abstractmethod
    def get_gravity(self):
        """
        Returns
        -------
        g (np.array): Gravity forces in generalized coordinate
        """
        pass

    @abc.abstractmethod
    def get_coriolis(self):
        """
        Returns
        -------
        c (np.array): Coriolis forces in generalized coordinate
        """
        pass

    @abc.abstractmethod
    def get_com_pos(self):
        """
        Returns
        -------
        com_pos (np.array): COM position
        """
        pass

    @abc.abstractmethod
    def get_com_lin_vel(self):
        """
        Returns
        -------
        com_lin_vel (np.array): COM linear velocity
        """
        pass

    @abc.abstractmethod
    def get_com_lin_jacobian(self):
        """
        Returns
        -------
        com_lin_jac (np.array): COM linear jacobian
        """
        pass

    @abc.abstractmethod
    def get_com_lin_jacobian_dot(self):
        """
        Returns
        -------
        com_lin_jac_dot (np.array): COM linear jacobian dot
        """
        pass

    @abc.abstractmethod
    def get_link_iso(self, link_id):
        """
        Parameters
        ----------
        link_id (str):
            Link ID
        Returns
        -------
        link_iso (np.array): Link SE(3)
        """
        pass

    @abc.abstractmethod
    def get_link_vel(self, link_id):
        """
        Parameters
        ----------
        link_id (str):
            Link ID
        Returns
        -------
            Link CoM Screw described in World Frame
        """
        pass

    @abc.abstractmethod
    def get_link_jacobian(self, link_id):
        """
        Link CoM Jacobian described in World Frame

        Parameters
        ----------
        link_id (str):
            Link ID
        Returns
        -------
        Jacobian (np.ndarray):
            Link CoM Jacobian described in World Frame
        """
        pass

    @abc.abstractmethod
    def get_link_jacobian_dot_times_qdot(self, link_id):
        """
        Parameters
        ----------
        link_id (str):
            Link ID
        Returns
        -------
            Link CoM jacobian_dot times qdot
        """
        pass

    @abc.abstractmethod
    def get_Ag(self):
        """
        Returns
        -------
        Ag (np.array)
        """
        pass

    @abc.abstractmethod
    def get_Ig(self):
        """
        Returns
        -------
        Ig (np.array)
        """
        pass

    @abc.abstractmethod
    def get_hg(self):
        """
        Returns
        -------
        hg (np.array)
        """
        pass
