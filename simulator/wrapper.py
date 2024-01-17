import copy

import gym
import numpy as np
from scipy.spatial.transform import Rotation as R

OBS_EEF_KEYS = [
    "rh_eef_pos",
    "lh_eef_pos",
    "rf_foot_pos",
    "lf_foot_pos",
    "rh_eef_quat",
    "lh_eef_quat",
    "rf_foot_quat",
    "lf_foot_quat",
]

OBS_JOINT_KEYS = [
    "r_hip_ie",
    "r_hip_aa",
    "r_hip_fe",
    "r_knee_fe_jp",
    "r_knee_fe_jd",
    "r_ankle_fe",
    "r_ankle_ie",
    "l_hip_ie",
    "l_hip_aa",
    "l_hip_fe",
    "l_knee_fe_jp",
    "l_knee_fe_jd",
    "l_ankle_fe",
    "l_ankle_ie",
    "r_shoulder_fe",
    "r_shoulder_aa",
    "r_shoulder_ie",
    "r_elbow_fe",
    "r_wrist_ps",
    "r_wrist_pitch",
    "l_shoulder_fe",
    "l_shoulder_aa",
    "l_shoulder_ie",
    "l_elbow_fe",
    "l_wrist_ps",
    "l_wrist_pitch",
    "neck_pitch",
]


def wrap_action(action_cur, action_prv):
    # PRV DATA: left<>right

    trajectory_cur = action_cur["trajectory"]
    trajectory_prv = action_prv["trajectory"]

    gripper = action_cur["gripper"]
    locomotion = action_cur["locomotion"]

    act_trajecory_right_delta_pos = (
        trajectory_cur["right_pos"] - trajectory_prv["right_pos"]
    )
    act_trajecory_left_delta_pos = (
        trajectory_cur["left_pos"] - trajectory_prv["left_pos"]
    )

    act_trajecory_right_rot_cur = R.from_quat(trajectory_cur["right_quat"]).as_matrix()
    act_trajecory_left_rot_cur = R.from_quat(trajectory_cur["left_quat"]).as_matrix()

    act_trajecory_right_rot_prv = R.from_quat(trajectory_prv["right_quat"]).as_matrix()
    act_trajecory_left_rot_prv = R.from_quat(trajectory_prv["left_quat"]).as_matrix()

    act_trajecory_right_delta_rot = act_trajecory_right_rot_cur @ (
        act_trajecory_right_rot_prv.transpose()
    )
    act_trajecory_left_delta_rot = act_trajecory_left_rot_cur @ (
        act_trajecory_left_rot_prv.transpose()
    )

    act_trajecory_right_delta_quat = R.from_matrix(
        act_trajecory_right_delta_rot
    ).as_quat()
    act_trajecory_left_delta_quat = R.from_matrix(
        act_trajecory_left_delta_rot
    ).as_quat()

    act_descrete = np.array([locomotion, gripper["right"], gripper["left"]])
    act_trajecory = np.concatenate(
        [
            act_trajecory_right_delta_pos,
            act_trajecory_left_delta_pos,
            act_trajecory_right_delta_quat,
            act_trajecory_left_delta_quat,
        ]
    )

    return np.concatenate([act_descrete, act_trajecory])


def wrap_obs(obs):
    pass


class EnvWrapper(gym.Env):
    def __init__(self, env) -> None:
        self._env = env
        self._renderer_right = None
        self._renderer_left = None
        self.sim = self._env.sim

    def reset(self, sim_reset=True, **kwargs):
        self._subtask = 0

        if sim_reset:
            obs = self._env.reset(**kwargs)
        else:
            obs = self._env._get_obs()

        self._cur_obs = {}
        self._cur_action = copy.deepcopy(self._env._cur_action)

        self._cur_obs.update(
            {
                "discrete_action": np.array([0, 0]),
                "trajectory_action": np.array(
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
                ),
            }
        )
        self._update_obs(obs)

        return self._get_obs()

    def step(self, action=None):
        self._update_action(action)
        obs = self._env.step(self._get_action())
        self._update_obs(obs)

        return self._get_obs(), 0, self.done, {"subtask": self.subtask}

    def _update_action(self, action):
        act_locomotion = np.round(action[0])
        # act_gripper_left = np.round(action[1])
        # act_gripper_right = np.round(action[2])
        if action[1] > 0.55:
            act_gripper_right = 1
        else:
            act_gripper_right = 0
        if action[2] > 0.55:
            act_gripper_left = 1
        else:
            act_gripper_left = 0
        act_right_pos = action[3:6]
        act_left_pos = action[6:9]
        act_right_quat = action[9:13]
        act_left_quat = action[13:17]
        # act_right_euler = action[9:12]
        # act_left_euler = action[12:15]

        self._cur_obs.update(
            {
                "discrete_action": np.copy(action[1:3]),
                "trajectory_action": np.copy(action[3:]),
            }
        )

        self._cur_action["locomotion"] = act_locomotion
        self._cur_action["gripper"]["right"] = act_gripper_right
        self._cur_action["gripper"]["left"] = act_gripper_left
        # self._cur_action['aux']['neck'] += act_aux_neck

        self._cur_action["trajectory"]["right_pos"] += act_right_pos
        self._cur_action["trajectory"]["left_pos"] += act_left_pos

        act_trajecory_right_rot = (
            R.from_quat(self._cur_action["trajectory"]["right_quat"]).as_matrix()
            @ R.from_quat(act_right_quat).as_matrix()
        )
        act_trajecory_left_rot = (
            R.from_quat(self._cur_action["trajectory"]["left_quat"]).as_matrix()
            @ R.from_quat(act_left_quat).as_matrix()
        )

        # act_trajecory_right_rot = R.from_quat(self._cur_action['trajectory']['right_quat']).as_matrix() @ R.from_euler('xyz', act_right_euler).as_matrix()
        # act_trajecory_left_rot = R.from_quat(self._cur_action['trajectory']['left_quat']).as_matrix() @ R.from_euler('xyz', act_left_euler).as_matrix()

        self._cur_action["trajectory"]["right_quat"] = R.from_matrix(
            act_trajecory_right_rot
        ).as_quat()
        self._cur_action["trajectory"]["left_quat"] = R.from_matrix(
            act_trajecory_left_rot
        ).as_quat()

    def _get_obs(self):
        return copy.deepcopy(self._cur_obs)

    def _get_action(self):
        return copy.deepcopy(self._cur_action)

    def _update_obs(self, obs):
        obs_trajectory = {
            eef_key: np.array(obs["trajectory"][eef_key]) for eef_key in OBS_EEF_KEYS
        }
        self._cur_obs.update(obs_trajectory)

        obs_joint_pos = np.array(
            [obs["joint_pos"][joint_key] for joint_key in OBS_JOINT_KEYS]
        )
        obs_joint_vel = np.array(
            [obs["joint_vel"][joint_key] for joint_key in OBS_JOINT_KEYS]
        )
        self._cur_obs.update(
            {
                "joint": np.concatenate(
                    (np.cos(obs_joint_pos), np.sin(obs_joint_pos), obs_joint_vel),
                    axis=0,
                )
            }
        )

        self._cur_obs.update({"state": np.array(obs["state"], dtype=np.uint8)})

        obs_rgbs = self._get_stereo()
        self._cur_obs.update(
            {
                "right_rgb": np.array(obs_rgbs["right"], dtype=np.uint8).transpose(
                    2, 0, 1
                ),
                "left_rgb": np.array(obs_rgbs["left"], dtype=np.uint8).transpose(
                    2, 0, 1
                ),
            }
        )

    def set_stereo_renderer(self, renderer_right, renderer_left):
        self._renderer_right = renderer_right
        self._renderer_left = renderer_left

    def set_renderer(self, renderer):
        self._env.set_renderer(renderer)

    def close(self):
        if self._env.renderer == None:
            self._env.renderer.close()
        if self._renderer_left == None:
            self._renderer_left.close()
        if self._renderer_right == None:
            self._renderer_right.close()

    def _render(self):
        if self._env.renderer == None:
            return
        else:
            return self._env.render()

    def _get_stereo(self):
        if self._renderer_right == None or self._renderer_left == None:
            return
        else:
            return {
                "right": self._renderer_right.render(),
                "left": self._renderer_left.render(),
            }

    @property
    def cur_time(self):
        return self._env.cur_time

    @property
    def done(self):
        return self._env.done

    @property
    def subtask(self):
        return self._env.subtask

    @property
    def success(self):
        return self._env.success


if __name__ == "__main__":
    pass
