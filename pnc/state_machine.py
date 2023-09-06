import abc


class StateMachine(abc.ABC):
    def __init__(self, state_id, robot):
        """
        Parameters
        ----------
        state_id (int): State id
        robot (RobotSystem)
        """
        self._robot = robot
        self._state_machine_time = 0.
        self._state_id = state_id

    @property
    def state_id(self):
        return self._state_id

    @abc.abstractmethod
    def one_step(self):
        pass

    @abc.abstractmethod
    def first_visit(self):
        pass

    @abc.abstractmethod
    def last_visit(self):
        pass

    @abc.abstractmethod
    def end_of_state(self):
        pass

    @abc.abstractmethod
    def get_next_state(self):
        pass
