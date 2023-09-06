import argparse
import numpy as np
import copy
import time
import zmq
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

# the comments are to appease pep8's complaints about import order
from util import geom
from simulator.envs import DoorEnv, EmptyEnv
from simulator.render import VRRenderer, getVRPose
from simulator.recorder import HDF5Recorder

RIGHTFORWARD_GRIPPER = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
TRANSFORM_VR = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

MEAN_INIT_POS = np.array([-0.5, 0.0, 0.743])
STD_INIT_POS = np.array([0.1, 0.2, 0.0])
MEAN_INIT_YAW = 0.0
STD_INIT_YAW = 0.2

ENV_LOOKUP = {
    'door': DoorEnv,
}

def main(env_type, subtask, demonstrator, host):

    # zmq config
    context = zmq.Context()

    control_socket = context.socket(zmq.PULL)
    control_socket.set(zmq.CONFLATE, 1)
    control_socket.connect("tcp://" + host + ":5555")
    print("first socket connected")

    video_socket = context.socket(zmq.PUSH)
    video_socket.set(zmq.CONFLATE, 1)
    video_socket.bind("tcp://*:5556")
    print("second socket connected")

    if env_type in ENV_LOOKUP.keys():
        env_class = ENV_LOOKUP[env_type]
    else:
        env_class = EmptyEnv

    env = env_class()
    env.config['Manipulation']['Trajectory Mode'] = 'interpolation'

    renderer = VRRenderer(socket=video_socket, sim=env.sim,
                          cam_name=env.robot.naming_prefix+'robotview')

    env.set_renderer(renderer)

    recorder = HDF5Recorder(sim=env.sim, config=env.config, file_path='./datasets/{}/subtask{}_{}/{}'.format(
        env_type, subtask, demonstrator, int(time.time())))
    env.set_recorder(recorder)
    env.reset(subtask=subtask)

    filtered_trajectory_action = copy.deepcopy(
        env.config['Action']['Default']['trajectory'])
    init_cnt = 0
    subtask_button_held = False

    while True:

        action = {}
        action['trajectory'] = {}
        action['locomotion'] = 0
        action['subtask'] = 0
        action['gripper'] = {}
        action['aux'] = {}

        if env_type == 'kitchen' or env_type == 'workbench2':
            action['aux']['neck'] = 1

        lh_target_pos, rh_target_pos, left_orientation, right_orientation, left_trigger, left_bump, left_button, left_pad, right_trigger, right_bump, right_button, right_pad = getVRPose(
            control_socket)

        rh_target_rot = np.dot(RIGHTFORWARD_GRIPPER, right_orientation)
        lh_target_rot = np.dot(RIGHTFORWARD_GRIPPER, left_orientation)

        if init_cnt < 10:
            action['trajectory'].update(filtered_trajectory_action)
            init_cnt += 1

        else:
            action['trajectory']['right_pos'] = np.copy(rh_target_pos)
            action['trajectory']['left_pos'] = np.copy(lh_target_pos)
            action['trajectory']['right_quat'] = geom.rot_to_quat(
                rh_target_rot)
            action['trajectory']['left_quat'] = geom.rot_to_quat(lh_target_rot)

        if left_button == 1:
            action['locomotion'] = 1

        elif right_button == 1:
            action['locomotion'] = 3
        elif right_pad == 1:
            action['locomotion'] = 4
        elif left_trigger == 1:
            action['locomotion'] = 5
        elif right_trigger == 1:
            action['locomotion'] = 6

        if left_bump == 1:
            action['gripper']['left'] = 1
        else:
            action['gripper']['left'] = 0
        if right_bump == 1:
            action['gripper']['right'] = 1
        else:
            action['gripper']['right'] = 0

        env.step(action)
        if left_pad == 1:
            if not subtask_button_held:
                subtask_button_held = True
                if not (bool(right_bump) ^ bool(left_bump)):
                    print("subtask segmented")
                    recorder.close()

                recorder = HDF5Recorder(sim=env.sim, config=env.config, file_path='./datasets/{}/subtask{}_{}/{}'.format(
                    env_type, subtask, demonstrator, int(time.time())))
                env.set_recorder(recorder)
                env.reset(subtask=subtask)
        else:
            subtask_button_held = False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='door',
                        help="")
    parser.add_argument("--subtask", type=int, default=0,
                        help="")
    parser.add_argument("--demonstrator", type=str, default='user',
                        help="")
    parser.add_argument("--host", type=str, default='192.168.50.50',
                        help="")
    args = parser.parse_args()

    env_type = args.env
    subtask = args.subtask
    demonstrator = args.demonstrator
    host = args.host

    main(env_type=env_type, subtask=subtask, demonstrator=demonstrator, host=host)
