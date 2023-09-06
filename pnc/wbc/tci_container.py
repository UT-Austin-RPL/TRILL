import abc


class TCIContainer(abc.ABC):
    def __init__(self, robot):
        self._robot = robot
        self._task_list = []
        self._contact_list = []
        self._internal_constraint_list = []

    @property
    def task_list(self):
        return self._task_list

    @property
    def contact_list(self):
        return self._contact_list

    @property
    def internal_constraint_list(self):
        return self._internal_constraint_list
