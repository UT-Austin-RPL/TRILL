"""
6-DoF gripper with its open/close variant
"""
import os

import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel

cwd = os.getcwd()

PATH_TO_GRIPPER_MODEL = os.path.expanduser(cwd + "/models/grippers/sake_ezgripper")
PATH_TO_GRIPPER_XML = os.path.join(PATH_TO_GRIPPER_MODEL, "sake_ezgripper.xml")

EZGRIPPER_MAP = {
    "joint": {
        "gripper": "joint_right_driver",
    },
    "actuator": {
        "gripper": "torque_drive",
    },
}


class SakeEZGripperBase(GripperModel):
    """
    6-DoF Robotiq gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(PATH_TO_GRIPPER_XML, idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0, 0, 0, 0, 0, 0, 0, 0])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["geom_left_finger1", "geom_left_finger2"],
            "right_finger": ["geom_right_finger1", "geom_right_finger2"],
            "left_fingerpad": ["geom_left_pad_box1_col", "geom_left_pad_box2_col"],
            "right_fingerpad": ["geom_right_pad_box1_col", "geom_right_pad_box2_col"],
        }


class SakeEZGripper(SakeEZGripperBase):
    """
    1-DoF variant of RobotiqGripperBase.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_key_map()

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = np.clip(
            self.current_action + self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    def _set_key_map(self):
        """
        Sets the key map for this gripper
        """

        self._key_map = {"joint": {}, "actuator": {}}

        for key in EZGRIPPER_MAP["joint"].keys():
            self._key_map["joint"].update(
                {key: self.naming_prefix + EZGRIPPER_MAP["joint"][key]}
            )
            self._key_map["actuator"].update(
                {key: self.naming_prefix + EZGRIPPER_MAP["actuator"][key]}
            )

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1

    @property
    def key_map(self):
        return self._key_map
