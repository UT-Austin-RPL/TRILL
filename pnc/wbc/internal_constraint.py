import abc
import numpy as np


class InternalConstraint(abc.ABC):
    """
    WBC Internal Constraint
    Usage:
        update_internal_constraint
    """
    def __init__(self, robot, dim):
        self._robot = robot
        self._dim = dim
        self._jacobian = np.zeros((self._dim, self._robot.n_q_dot))
        self._jacobian_dot_q_dot = np.zeros(self._dim)

    @property
    def jacobian(self):
        return self._jacobian

    @property
    def jacobian_dot_q_dot(self):
        return self._jacobian_dot_q_dot

    def update_internal_constraint(self):
        self._update_jacobian()

    @abc.abstractmethod
    def _update_jacobian(self):
        pass
