import numpy as np
from util import filters


class JointIntegrator(object):
    def __init__(self, n_joint, dt):
        self._n_joint = n_joint
        self._dt = dt

        self._pos_cutoff_freq = 0.
        self._vel_cutoff_freq = 0.
        self._max_pos_err = 0.
        self._joint_pos_limit = None
        self._joint_vel_limit = None

        self._pos = np.zeros(n_joint)
        self._vel = np.zeros(n_joint)

    @property
    def pos_cutoff_freq(self):
        return self._pos_cutoff_freq

    @property
    def vel_cutoff_freq(self):
        return self._vel_cutoff_freq

    @property
    def max_pos_err(self):
        return self._max_pos_err

    @property
    def joint_pos_limit(self):
        return self._joint_pos_limit

    @property
    def joint_vel_limit(self):
        return self._joint_vel_limit

    @pos_cutoff_freq.setter
    def pos_cutoff_freq(self, val):
        self._pos_cutoff_freq = val
        self._alpha_pos = filters.get_alpha_from_frequency(
            self._pos_cutoff_freq, self._dt)

    @vel_cutoff_freq.setter
    def vel_cutoff_freq(self, val):
        self._vel_cutoff_freq = val
        self._alpha_vel = filters.get_alpha_from_frequency(
            self._vel_cutoff_freq, self._dt)

    @max_pos_err.setter
    def max_pos_err(self, val):
        self._max_pos_err = val

    @joint_pos_limit.setter
    def joint_pos_limit(self, val):
        assert val.shape[0] == self._n_joint
        self._joint_pos_limit = val

    @joint_vel_limit.setter
    def joint_vel_limit(self, val):
        assert val.shape[0] == self._n_joint
        self._joint_vel_limit = val

    def initialize_states(self, init_vel, init_pos):
        assert init_vel.shape[0] == self._n_joint
        assert init_pos.shape[0] == self._n_joint

        self._pos = init_pos
        self._vel = init_vel

    def integrate(self, acc, vel, pos):
        self._vel = np.clip(
            (1.0 - self._alpha_vel) * self._vel + acc * self._dt,
            self._joint_vel_limit[:, 0], self._joint_vel_limit[:, 1])
        self._pos = np.clip((1.0 - self._alpha_pos) * self._pos +
                            self._alpha_pos * pos + self._vel * self._dt,
                            self._joint_pos_limit[:,
                                                  0], self._joint_pos_limit[:,
                                                                            1])
        return self._vel, self._pos
