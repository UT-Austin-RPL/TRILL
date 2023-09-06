import abc
from collections import OrderedDict


class ControlArchitecture(abc.ABC):
    def __init__(self, robot):
        self._robot = robot

        self._state = 0
        self._prev_state = 0
        self._b_state_first_visit = True

        self._state_machine = OrderedDict()
        self._trajectory_managers = OrderedDict()
        self._hierarchy_manangers = OrderedDict()
        self._reaction_force_managers = OrderedDict()

    @property
    def state(self):
        return self._state

    @property
    def prev_state(self):
        return self._prev_state

    @abc.abstractmethod
    def get_command(self):
        pass
