import h5py
import pickle
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

from simulator.wrapper import EnvWrapper, wrap_action, wrap_obs
from simulator.envs import DoorEnv, EmptyEnv
from simulator.recorder import HDF5Player
from simulator.render import CV2Renderer
import argparse
import numpy as np
from util import geom
from mimic import policy_from_checkpoint

MEAN_INIT_POS = [-0.5, 0.0, 0.743]
STD_INIT_POS = [0, 0.2, 0.0]
MEAN_INIT_YAW = 0.0
STD_INIT_YAW = 0.2

ENV_LOOKUP = {
    'door': DoorEnv,
}


def main(**kwargs):
    evaluate(**kwargs)

def evaluate(path, gui, env_type, cam_name='upview', subtask=0, seed=0, save_path=None,  **kwargs):

    if env_type in ENV_LOOKUP.keys():
        _env = ENV_LOOKUP[env_type]()

    else:
        _env = EmptyEnv()

    eval_policy = policy_from_checkpoint(ckpt_path=path)[0]
    renderer = CV2Renderer(device_id=-1, sim=_env.sim, width=400, height=300, cam_name=cam_name, gui=gui,
                           save_path=save_path)

    np.random.seed(seed)

    env = EnvWrapper(_env)
    env.set_renderer(renderer)
    renderer_right = CV2Renderer(
        device_id=-1, sim=env.sim, width=240, height=180, cam_name='robot0_robotright', gui=False)
    renderer_left = CV2Renderer(
        device_id=-1, sim=env.sim, width=240, height=180, cam_name='robot0_robotleft', gui=False)
    env.set_stereo_renderer(renderer_right=renderer_right,
                            renderer_left=renderer_left)

    obs = env.reset(subtask=subtask)
    obs['right_rgb'] = obs['right_rgb']/255.0
    obs['left_rgb'] = obs['left_rgb']/255.0
    
    done = False
    log_time = env.cur_time + 1.0

    if env_type == 'workbench2' and subtask == 1:

        init_time = env.cur_time
        init_pos = np.array([0.22,-0.25, 0.1 ])
        target_pos = np.array([0.22,-0.35, -0.0 ]) #+ np.random.normal(0, 0.02, size=3)

        RIGHTFORWARD_GRIPPER = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

        while env.cur_time < 5.0:

            env_traj_action = {}
            pos_period = 4.0
            quat_period = 5.0

            if env.cur_time < pos_period:
                pos_phase = (env.cur_time-init_time)/(pos_period - init_time)
            else:
                pos_phase = 1.0
            if env.cur_time < quat_period:
                quat_phase = (env.cur_time-init_time)/(quat_period - init_time)
            else:
                quat_phase = 1.0
            lh_input = geom.euler_to_rot(np.pi/180*np.array([0, 0, 0]))
            rh_input = geom.euler_to_rot(-0.6*np.pi*np.sin(0.5*np.pi*quat_phase)*np.array([1, 0, 0]))
            rh_target_pos = init_pos + (target_pos - init_pos)*np.sin(0.5*np.pi*pos_phase)

            rh_target_rot = np.dot(rh_input, RIGHTFORWARD_GRIPPER)
            lh_target_rot = np.dot(lh_input, RIGHTFORWARD_GRIPPER)

            env_traj_action['right_pos'] = rh_target_pos
            env_traj_action['right_quat'] = geom.rot_to_quat(rh_target_rot)
            env_traj_action['left_quat'] = geom.rot_to_quat(lh_target_rot)

            env._cur_action['trajectory'].update(env_traj_action)
            env_obs = env._env.step(env._get_action())
            env._update_obs(env_obs)
            obs = env._get_obs()

            obs['right_rgb'] = obs['right_rgb']/255.0
            obs['left_rgb'] = obs['left_rgb']/255.0


    while not done:
        action = np.array(eval_policy(obs))
        # action = np.concatenate(([0], eval_policy(obs)), axis=0)
        obs, _, _, _ = env.step(action)
        obs['right_rgb'] = obs['right_rgb']/255.0
        obs['left_rgb'] = obs['left_rgb']/255.
        # buffer.append(np.copy(action[1:3]))
        
        if env.cur_time > log_time:
            log_time = env.cur_time + 1.0
            # print("Processing...\tSubtask: {},\tSeed: {},\tTime: {},\tDoor: {}\tSave to:{}".format(subtask, seed, env.cur_time, env._env.door_angle, save_path))
            print("Processing...\tSubtask: {},\tSeed: {},\tTime: {},\tSave to:{}".format(subtask, seed, env.cur_time, save_path))

        if env.success or env.cur_time>70 or env._env.robot_pos[2] < 0.3:
            done = True

        if env.cur_time>20 and subtask == 1:
            done = True

    renderer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None,
                        help="")
    parser.add_argument("--gui", type=int, default=1,
                        help="")
    parser.add_argument("--env", type=str, default='door',
                        help="")
    parser.add_argument("--cam", type=str, default='upview',
                        help="")
    parser.add_argument("--subtask", type=int, default=0,
                        help="")
    parser.add_argument("--seed", type=int, default=None,
                        help="")
    parser.add_argument("--device", type=str, default='cuda',
                        help="")
    parser.add_argument("--save", type=str, default=None,
                        help="")
    args = parser.parse_args()

    path = args.path
    gui = args.gui
    env_type = args.env
    cam_name = args.cam
    subtask = args.subtask
    device = args.device
    seed = args.seed

    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        save_path = os.path.join(args.save, '{}_{}_{}_{}.mp4'.format(seed, env_type, cam_name, subtask))
    else:
        save_path = None

    main(path=path, gui=gui, env_type=env_type, cam_name=cam_name,
         subtask=subtask, save_path=save_path, device=device, seed=seed)
