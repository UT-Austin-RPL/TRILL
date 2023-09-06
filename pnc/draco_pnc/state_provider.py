import numpy as np
from collections import OrderedDict

class MetaSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton,
                                        cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DracoManipulationStateProvider(metaclass=MetaSingleton):
    def __init__(self, robot):
        self._robot = robot
        # self._nominal_joint_pos = OrderedDict()
        # self._state = 0
        # self._prev_state = 0
        # self._curr_time = 0.
        # self._dcm = np.zeros(3)
        # self._prev_dcm = np.zeros(3)
        # self._dcm_vel = np.zeros(3)
        # self._b_rf_contact = True
        # self._b_lf_contact = True
        self.initialize()

    def initialize(self):
        self._nominal_joint_pos = OrderedDict()
        self._state = 0
        self._prev_state = 0
        self._curr_time = 0.
        self._dcm = np.zeros(3)
        self._prev_dcm = np.zeros(3)
        self._dcm_vel = np.zeros(3)
        self._b_rf_contact = True
        self._b_lf_contact = True

    @property
    def nominal_joint_pos(self):
        return self._nominal_joint_pos

    @property
    def state(self):
        return self._state

    @property
    def prev_state(self):
        return self._prev_state

    @property
    def dcm(self):
        return self._dcm

    @dcm.setter
    def dcm(self, value):
        self._dcm = value

    @property
    def prev_dcm(self):
        return self._prev_dcm

    @prev_dcm.setter
    def prev_dcm(self, value):
        self._prev_dcm = value

    @property
    def dcm_vel(self):
        return self._dcm_vel

    @dcm_vel.setter
    def dcm_vel(self, value):
        self._dcm_vel = value

    @prev_state.setter
    def prev_state(self, value):
        self._prev_state = value

    @property
    def curr_time(self):
        return self._curr_time

    @nominal_joint_pos.setter
    def nominal_joint_pos(self, val):
        assert self._robot.n_a == len(val.keys())
        self._nominal_joint_pos = val

    @state.setter
    def state(self, val):
        self._state = val

    @curr_time.setter
    def curr_time(self, val):
        self._curr_time = val

    @property
    def b_rf_contact(self):
        return self._b_rf_contact

    @b_rf_contact.setter
    def b_rf_contact(self, value):
        self._b_rf_contact = value

    @property
    def b_lf_contact(self):
        return self._b_lf_contact

    @b_lf_contact.setter
    def b_lf_contact(self, value):
        self._b_lf_contact = value
