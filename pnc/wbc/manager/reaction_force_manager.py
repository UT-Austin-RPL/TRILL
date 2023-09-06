import numpy as np


class ReactionForceManager(object):
    def __init__(self, contact, maximum_rf_z_max):
        self._contact = contact
        self._maximum_rf_z_max = maximum_rf_z_max
        self._minimum_rf_z_max = 0.001
        self._starting_rf_z_max = contact.rf_z_max
        self._start_time = 0.
        self._duration = 0.

    def initialize_ramp_to_min(self, start_time, duration):
        self._start_time = start_time
        self._duration = duration
        self._starting_rf_z_max = self._contact.rf_z_max

    def initialize_ramp_to_max(self, start_time, duration):
        self._start_time = start_time
        self._duration = duration
        self._starting_rf_z_max = self._contact.rf_z_max

    def update_ramp_to_min(self, current_time):
        t = np.clip(current_time, self._start_time,
                    self._start_time + self._duration)
        self._contact.rf_z_max = (
            self._minimum_rf_z_max - self._starting_rf_z_max
        ) / self._duration * (t - self._start_time) + self._starting_rf_z_max

    def update_ramp_to_max(self, current_time):
        t = np.clip(current_time, self._start_time,
                    self._start_time + self._duration)
        self._contact.rf_z_max = (
            self._maximum_rf_z_max - self._starting_rf_z_max
        ) / self._duration * (t - self._start_time) + self._starting_rf_z_max
