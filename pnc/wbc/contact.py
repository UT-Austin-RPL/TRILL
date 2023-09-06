import abc
import numpy as np


class Contact(abc.ABC):
    """
    WBC Contact
    -----------
    Usage:
        update_contact
    """
    def __init__(self, robot, dim):
        self._robot = robot
        self._dim_contact = dim
        self._jacobian = np.zeros((self._dim_contact, self._robot.n_q))
        self._jacobian_dot_q_dot = np.zeros(self._dim_contact)
        self._rf_z_max = 0.
        self._cone_constraint_mat = None
        self._cone_constraint_vec = None

    @property
    def jacobian(self):
        return self._jacobian

    @property
    def jacobian_dot_q_dot(self):
        return self._jacobian_dot_q_dot

    @property
    def dim_contact(self):
        return self._dim_contact

    @property
    def rf_z_max(self):
        return self._rf_z_max

    @rf_z_max.setter
    def rf_z_max(self, value):
        if value <= 0.:
            value = 1e-3
        self._rf_z_max = value

    @property
    def cone_constraint_mat(self):
        return self._cone_constraint_mat

    @property
    def cone_constraint_vec(self):
        return self._cone_constraint_vec

    def update_contact(self):
        self._update_jacobian()
        self._update_cone_constraint()

    @abc.abstractmethod
    def _update_jacobian(self):
        pass

    @abc.abstractmethod
    def _update_cone_constraint(self):
        pass
