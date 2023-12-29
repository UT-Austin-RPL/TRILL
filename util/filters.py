import numpy as np


def get_alpha_from_frequency(hz, dt):
    omega = 2 * np.pi * hz
    alpha = (omega * dt) / (1.0 + (omega * dt))

    return np.clip(alpha, 0.0, 1.0)
