import os

import numpy as np

from .mobile import HumanoidModel

cwd = os.getcwd()

PATH_TO_ROBOT_MODEL = os.path.expanduser(cwd + "/models/robots/draco3")
PATH_TO_ROBOT_XML = os.path.join(PATH_TO_ROBOT_MODEL, "robot.xml")

DRACO3_MAP = {
    "joint": {
        "l_shoulder_fe": "joint_left_S0",
        "l_shoulder_aa": "joint_left_S1",
        "l_shoulder_ie": "joint_left_S2",
        "l_elbow_fe": "joint_left_E0",
        "l_wrist_ps": "joint_left_W0",
        "l_wrist_pitch": "joint_left_W1",
        "r_shoulder_fe": "joint_right_S0",
        "r_shoulder_aa": "joint_right_S1",
        "r_shoulder_ie": "joint_right_S2",
        "r_elbow_fe": "joint_right_E0",
        "r_wrist_ps": "joint_right_W0",
        "r_wrist_pitch": "joint_right_W1",
        "l_hip_ie": "joint_left_H0",
        "l_hip_aa": "joint_left_H1",
        "l_hip_fe": "joint_left_H2",
        "l_knee_fe_jp": "joint_left_N0",
        "l_knee_fe_jd": "joint_left_N2",
        "l_ankle_fe": "joint_left_A0",
        "l_ankle_ie": "joint_left_A1",
        "r_hip_ie": "joint_right_H0",
        "r_hip_aa": "joint_right_H1",
        "r_hip_fe": "joint_right_H2",
        "r_knee_fe_jp": "joint_right_N0",
        "r_knee_fe_jd": "joint_right_N2",
        "r_ankle_fe": "joint_right_A0",
        "r_ankle_ie": "joint_right_A1",
        "neck_pitch": "joint_head",
    },
    "actuator": {
        "l_shoulder_fe": "torque_left_arm_0",
        "l_shoulder_aa": "torque_left_arm_1",
        "l_shoulder_ie": "torque_left_arm_2",
        "l_elbow_fe": "torque_left_arm_3",
        "l_wrist_ps": "torque_left_arm_4",
        "l_wrist_pitch": "torque_left_arm_5",
        "r_shoulder_fe": "torque_right_arm_0",
        "r_shoulder_aa": "torque_right_arm_1",
        "r_shoulder_ie": "torque_right_arm_2",
        "r_elbow_fe": "torque_right_arm_3",
        "r_wrist_ps": "torque_right_arm_4",
        "r_wrist_pitch": "torque_right_arm_5",
        "l_hip_ie": "torque_left_leg_0",
        "l_hip_aa": "torque_left_leg_1",
        "l_hip_fe": "torque_left_leg_2",
        "l_knee_fe_jp": "torque_left_leg_3",
        "l_knee_fe_jd": "torque_left_leg_4",
        "l_ankle_fe": "torque_left_leg_5",
        "l_ankle_ie": "torque_left_leg_6",
        "r_hip_ie": "torque_right_leg_0",
        "r_hip_aa": "torque_right_leg_1",
        "r_hip_fe": "torque_right_leg_2",
        "r_knee_fe_jp": "torque_right_leg_3",
        "r_knee_fe_jd": "torque_right_leg_4",
        "r_ankle_fe": "torque_right_leg_5",
        "r_ankle_ie": "torque_right_leg_6",
        "neck_pitch": "torque_head",
    },
}


class Draco3(HumanoidModel):
    """
    Panda is a sensitive single-arm robot designed by Franka.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(fname=PATH_TO_ROBOT_XML, idn=idn)

        # Set joint damping
        # self.set_joint_attribute(attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01)))

    # @property
    # def default_mount(self):
    #     return "Free"

    def _set_key_map(self):
        """
        Sets the key map for this robot
        """

        self._key_map = {"joint": {}, "actuator": {}}
        self._key_map["joint"].update(
            {
                key: self.naming_prefix + value
                for key, value in DRACO3_MAP["joint"].items()
            }
        )
        self._key_map["actuator"].update(
            {
                key: self.naming_prefix + value
                for key, value in DRACO3_MAP["actuator"].items()
            }
        )

    @property
    def default_gripper(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific gripper names
        """
        return {"right": "Robotiq85Gripper", "left": "Robotiq85Gripper"}

    # @property
    # def init_qpos(self):
    #     """
    #     Since this is bimanual robot, returns [right, left] array corresponding to respective values

    #     Note that this is a pose such that the arms are half extended

    #     Returns:
    #         np.array: default initial qpos for the right, left arms
    #     """
    #     # [right, left]
    #     # Arms half extended
    #     return np.array(
    #         [0.403, -0.636, 0.114, 1.432, 0.735, 1.205, -0.269, -0.403, -0.636, -0.114, 1.432, -0.735, 1.205, 0.269]
    #     )

    @property
    def default_controller_config(self):
        return "default_panda"

    @property
    def init_qpos(self):
        return np.array(
            [
                0,
                np.pi / 16.0,
                0.00,
                -np.pi / 2.0 - np.pi / 3.0,
                0.00,
                np.pi - 0.2,
                np.pi / 4,
            ]
        )

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific eef names
        """
        return {"right": "right_mount", "left": "left_mount"}
