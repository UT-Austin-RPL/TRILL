import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
from util import geom
from simulator.envs import DoorEnv, EmptyEnv
from simulator.render import CV2Renderer
from simulator.recorder import HDF5Recorder
import time
import argparse

RIGHTFORWARD_GRIPPER = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
TRANSFORM_VR =  np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]) #geom.euler_to_rot(np.array([0, np.pi, 0]))

ENV_LOOKUP = {
    'door': DoorEnv,
}


def main(gui, env_type, cam_name='upview', subtask=0, save_video=False):
    if env_type in ENV_LOOKUP.keys():
        env_class = ENV_LOOKUP[env_type]
    else:
        env_class = EmptyEnv

    env = env_class()

    env.config['Manipulation']['Trajectory Mode'] = 'interpolation'

    if save_video:
        save_path = os.path.join('.', '{}_{}_{}.mp4'.format(env_type, cam_name, subtask))
    else:
        save_path = None
    renderer = CV2Renderer(device_id=-1, sim=env.sim, cam_name=cam_name, gui=gui, save_path=save_path)
    recorder = None
    recorder = HDF5Recorder(sim=env.sim, config=env.config, file_path='./test/demo_{}_{}'.format(env_type, int(time.time())))

    env.set_renderer(renderer)
    env.set_recorder(recorder)
    env.reset(subtask=subtask)
    
    done = False
    subtask = 0

    init_time = env.cur_time

    right_pos = np.array([0.22,-0.25,0.1 ])
    left_pos = np.array([0.22, 0.25,0.1 ])
    target_pos = np.array([0.22,-0.35, -0.1 ])

    while not done:

        action = {}
        action['trajectory'] = {}
        action['gripper'] = {}
        action['aux'] = {}
        action['subtask'] = 0
        action['locomotion'] = 0

        rh_target_pos = right_pos
        lh_target_pos = left_pos
        lh_input = geom.euler_to_rot(np.array([0, 0, 0]))
        rh_input = geom.euler_to_rot(np.array([0, 0, 0]))

        if env.cur_time < 3.0 + init_time:
            if env.cur_time < 1.5 + init_time:
                phase = (env.cur_time-init_time)/6.0
            else:
                phase = 0.25
            lh_target_pos = left_pos + 0.2*np.sin(2*np.pi*phase)*np.array([-1.0, 0.0, -1.3])
            rh_target_pos = right_pos + 0.2*np.sin(2*np.pi*phase)*np.array([-1.0, 0.0, -1.3])
            action['locomotion'] = 0
        else:
            lh_target_pos = left_pos + 0.4*np.array([-0.5, 0.0, -1.0])
            rh_target_pos = right_pos + 0.4*np.array([-0.5, 0.0, -1.0])
            action['locomotion'] = 1            

        if env.cur_time < 6.0 + init_time:
            if env.cur_time < 1.5 + init_time:
                phase = (env.cur_time-init_time)/6.0
            elif env.cur_time < 2.0 + init_time:
                phase = 0.25
                action['gripper']['left'] = 0
            elif env.cur_time < 3.0 + init_time:
                phase = 0.25
                action['gripper']['left'] = 1
            lh_target_pos = left_pos + 0.1*np.sin(2*np.pi*phase)*np.array([0.85, -0.8, 0.5])
            rh_target_pos = right_pos + 0.1*np.sin(2*np.pi*phase)*np.array([0, 0, 0])

        elif env.cur_time < 12.0 + init_time:
            phase = (env.cur_time-init_time)/6.0
            lh_target_pos = left_pos + 0.1*np.array([0.85, -0.8, 0.5]) + 0.1*np.sin(2*np.pi*phase)*np.array([0, 0, -0.5])
            rh_target_pos = right_pos

        rh_target_rot = np.dot(rh_input, RIGHTFORWARD_GRIPPER)
        lh_target_rot = np.dot(lh_input, RIGHTFORWARD_GRIPPER)
        action['trajectory']['left_pos'] = lh_target_pos
        action['trajectory']['right_pos'] = rh_target_pos
        action['trajectory']['right_quat'] = geom.rot_to_quat(rh_target_rot)
        action['trajectory']['left_quat'] = geom.rot_to_quat(lh_target_rot)

        env.step(action)

        if env.cur_time > 24.0:
            done = True

    recorder.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=int, default=1,
                        help="")
    parser.add_argument("--env", type=str, default='door',
                        help="")
    parser.add_argument("--cam", type=str, default='upview',
                        help="")
    parser.add_argument("--subtask", type=int, default=1,
                        help="")
    args = parser.parse_args()

    gui = args.gui
    env_type = args.env
    cam_name = args.cam
    subtask = args.subtask

    main(gui=gui, env_type=env_type, cam_name=cam_name, subtask=subtask)
