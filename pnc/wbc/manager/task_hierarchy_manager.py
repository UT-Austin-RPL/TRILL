import numpy as np


class TaskHierarchyManager(object):
    def __init__(self, task, w_max, w_min):
        self._task = task
        self._w_max = w_max
        self._w_min = w_min
        self._w_starting = self._task.w_hierarchy
        self._start_time = 0.
        self._duration = 0.

    def initialize_ramp_to_min(self, start_time, duration):
        self._start_time = start_time
        self._duration = duration
        self._w_starting = self._task.w_hierarchy

    def initialize_ramp_to_max(self, start_time, duration):
        self._start_time = start_time
        self._duration = duration
        self._w_starting = self._task.w_hierarchy

    def update_ramp_to_min(self, current_time):
        t = np.clip(current_time, self._start_time,
                    self._start_time + self._duration)
        self._task.w_hierarchy = (self._w_min -
                                  self._w_starting) / self._duration * (
                                      t - self._start_time) + self._w_starting

    def update_ramp_to_max(self, current_time):
        t = np.clip(current_time, self._start_time,
                    self._start_time + self._duration)
        self._task.w_hierarchy = (self._w_max -
                                  self._w_starting) / self._duration * (
                                      t - self._start_time) + self._w_starting
