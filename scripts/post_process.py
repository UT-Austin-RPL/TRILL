import os
import sys
import argparse

cwd = os.getcwd()
sys.path.append(cwd)

from simulator.render import CV2Renderer
from simulator.recorder import HDF5Player
from simulator.wrapper import OBS_EEF_KEYS, OBS_JOINT_KEYS
from simulator.envs import DoorEnv, KitchenEnv, Kitchen2Env, EmptyEnv, WorkbenchEnv, Workbench2Env  # noqa: E402
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R
import collections

from simulator import sim_util
import ipdb


ENV_LOOKUP = {
    'door': DoorEnv,
}

def segment_fcn(player, mode="step"):
    
    if mode == "reset":
        player._cur_state = player._cur_obs['state']
        player._post_buffer = {'observation/subtask': collections.deque()}
        for label in OBS_EEF_KEYS:
            player._post_buffer['observation/trajectory/{}'.format(label)] = collections.deque()

    if mode == "step":
        if player._demo_count < player._demo_length:
            new_state = player._read_obs()['state']
            if (new_state == 1 and player._cur_state != 1) and player._subtask == 0:
                ipdb.set_trace()
            elif (new_state != 1 and player._cur_state == 1) and player._subtask == 1:                
                ipdb.set_trace()
            elif player._subtask == 2:
                pass
            player._cur_state = new_state

            player._post_buffer["observation/subtask"].append(player._subtask)
            trajectory = sim_util.get_trajectory(player._env.sim, player._env.robot)
            for key in trajectory.keys():
                player._post_buffer['observation/trajectory/{}'.format(key)].append(np.copy(trajectory[key]))

    elif mode == "close":
        os.system('cp {} {}'.format(
            os.path.join(player._path, 'demo.hdf5'), os.path.join(player._path, 'demo_old.hdf5')))  

        new_demo_file = h5py.File(os.path.join(player._path, 'demo.hdf5'), 'a')

        for label, buffer in player._post_buffer.items():
            assert len(buffer) == new_demo_file['time'].shape[0]
            if label in ['observation/trajectory/{}'.format(key) for key in ['lf_foot_pos', 'rf_foot_pos']]:
                del new_demo_file[label]
            new_demo_file.create_dataset(label, data=np.array(buffer), compression="gzip", chunks=True, dtype="f")

        new_demo_file.close()
            


def process_fcn(player, mode="step", dataname='data'):

    if mode == "reset":
        player._post_buffer = {'right_rgb': collections.deque(),
                               'left_rgb': collections.deque(),
                               }

        ### EXTENDING LOCMOTION COMMANDS ###
        player._locomotion_buffer = []
        player._walking_cnt = 0
        player._prv_locomotion = 0
        player._prv_state = 1

    if mode == "step":

        if player._demo_count < player._demo_length:

            ### EXTENDING LOCMOTION COMMANDS ###
            new_state = player._read_obs()['state']
            new_locomtion = player._read_action()['locomotion']

            player._prv_locomotion = new_locomtion
            player._locomotion_buffer.append(player._prv_locomotion)

            imgs = player._get_stereo()
            player._post_buffer['right_rgb'].append(np.copy(imgs['right']))
            player._post_buffer['left_rgb'].append(np.copy(imgs['left']))


    elif mode == "close":

        original_demo_file = h5py.File(os.path.join(player._path, 'demo.hdf5'), 'r')
        new_demo_file = h5py.File(os.path.join(player._path, 'demo_{}.hdf5'.format(dataname)), 'a')

        new_demo_file.create_dataset('subtask', data=np.round(original_demo_file['observation/subtask']), 
                                        compression="gzip", chunks=True, dtype='uint8')
        obs_group = new_demo_file.create_group("obs") 

        for label, buffer in player._post_buffer.items():
            obs_group.create_dataset(label, data=np.array(buffer), 
                                    compression="gzip", chunks=True, dtype='uint8')
        for eef_key in OBS_EEF_KEYS:
            obs_group.create_dataset(eef_key, data=np.array(original_demo_file['observation/trajectory/{}'.format(eef_key)]),
                                        compression="gzip", chunks=True, dtype='f')
        obs_joint_pos = np.column_stack([original_demo_file['observation/joint_pos/{}'.format(joint_key)] for joint_key in OBS_JOINT_KEYS])
        obs_joint_vel = np.column_stack([original_demo_file['observation/joint_vel/{}'.format(joint_key)] for joint_key in OBS_JOINT_KEYS])
        obs_group.create_dataset('joint', 
                                data=np.concatenate((np.cos(obs_joint_pos), np.sin(obs_joint_pos), obs_joint_vel), axis=1), 
                                compression="gzip", chunks=True, dtype='f')
        obs_group.create_dataset('state', data=np.array(original_demo_file['observation/state'])/15.0, 
                                compression="gzip", chunks=True, dtype='f')


        # extention for locomotion
        locomtion_buffer = np.array(player._locomotion_buffer)
        act_discrete = np.column_stack([locomtion_buffer, original_demo_file['action/gripper/right'], original_demo_file['action/gripper/left']])

        act_trajecory_right_pos = original_demo_file['action/trajectory/right_pos']
        act_trajecory_left_pos = original_demo_file['action/trajectory/left_pos']
        act_trajecory_right_quat =  original_demo_file['action/trajectory/right_quat']
        act_trajecory_left_quat = original_demo_file['action/trajectory/left_quat']

        act_trajecory_right_delta_pos = np.copy(act_trajecory_right_pos)
        act_trajecory_right_delta_pos[1:] -= act_trajecory_right_pos[:-1]
        act_trajecory_right_delta_pos[0] -= act_trajecory_right_pos[0]
        act_trajecory_left_delta_pos = np.copy(act_trajecory_left_pos)
        act_trajecory_left_delta_pos[1:] -= act_trajecory_left_pos[:-1]
        act_trajecory_left_delta_pos[0] -= act_trajecory_left_pos[0]

        act_trajecory_right_rot = R.from_quat(act_trajecory_right_quat).as_matrix()
        act_trajecory_right_delta_rot = np.copy(act_trajecory_right_rot)
        act_trajecory_right_delta_rot[1:] = act_trajecory_right_delta_rot[1:] @ (act_trajecory_right_rot[:-1].transpose(0,2,1))
        act_trajecory_right_delta_rot[0] = act_trajecory_right_delta_rot[0] @ (act_trajecory_right_rot[0].transpose())
        act_trajecory_right_delta_quat = R.from_matrix(act_trajecory_right_delta_rot).as_quat()
        act_trajecory_left_rot = R.from_quat(act_trajecory_left_quat).as_matrix()
        act_trajecory_left_delta_rot = np.copy(act_trajecory_left_rot)
        act_trajecory_left_delta_rot[1:] = act_trajecory_left_delta_rot[1:] @ (act_trajecory_left_rot[:-1].transpose(0,2,1))
        act_trajecory_left_delta_rot[0] = act_trajecory_left_delta_rot[0] @ (act_trajecory_left_rot[0].transpose())
        act_trajecory_left_delta_quat = R.from_matrix(act_trajecory_left_delta_rot).as_quat()
        act_trajecory = np.column_stack([act_trajecory_right_delta_pos, act_trajecory_left_delta_pos, act_trajecory_right_delta_quat, act_trajecory_left_delta_quat])

        act_concat = np.concatenate((act_discrete, act_trajecory), axis=1)
        obs_action = np.zeros(act_concat.shape)
        obs_action[1:] = act_concat[:-1]
        obs_action[0] = act_concat[0]

        obs_group.create_dataset('action', data=obs_action,
                                compression="gzip", chunks=True, dtype='f')

        new_demo_file.create_dataset('actions', 
                                    data=act_concat, 
                                    compression="gzip", chunks=True, dtype='f')
        original_demo_file.close()
        new_demo_file.close()


def segmentation(path, env_type="door"):
    
    if env_type in ENV_LOOKUP.keys():
        env = ENV_LOOKUP[env_type]()
    else:
        env = EmptyEnv()

    cam_name='robot0_replayview'
    player = HDF5Player(env, mode='replay', post_fcn=segment_fcn, file_path=path)
    renderer = CV2Renderer(device_id=-1, sim=env.sim, width=400, height=300, cam_name=cam_name, gui=True)
    renderer_right = CV2Renderer(device_id=-1, sim=env.sim, width=240, height=180, cam_name='robot0_robotright', gui=False)
    renderer_left = CV2Renderer(device_id=-1, sim=env.sim, width=240, height=180, cam_name='robot0_robotleft', gui=False)

    player.set_renderer(renderer)
    player.set_stereo_renderer(renderer_right=renderer_right, renderer_left=renderer_left)

    player.reset()

    while not player.done:
        player.step()
    player.close()
    renderer.close()
    renderer_right.close()
    renderer_left.close()


def post_process(path, env_type="door", dataname="data"):

    def post_process_fcn(player, mode="step"):
        return process_fcn(player, mode, dataname=dataname)

    if env_type in ENV_LOOKUP.keys():
        env = ENV_LOOKUP[env_type]()
    else:
        env = EmptyEnv()

    player = HDF5Player(env, mode='replay', post_fcn=post_process_fcn, file_path=path)
    renderer_right = CV2Renderer(device_id=-1, sim=env.sim, width=240, height=180, cam_name='robot0_robotright', gui=False)
    renderer_left = CV2Renderer(device_id=-1, sim=env.sim, width=240, height=180, cam_name='robot0_robotleft', gui=False)
    player.set_stereo_renderer(renderer_right=renderer_right, renderer_left=renderer_left)

    player.reset()

    while not player.done:
        player.step()
    player.close()
    renderer_right.close()
    renderer_left.close()


def batch_subtask_data(path, env_type="door", subtask=0, dataname="data"):
    out_file = h5py.File(os.path.join(path, '{}_{}_{}.hdf5').format(dataname, env_type, subtask), 'w')
    out_data = out_file.create_group("data")

    demo_file_paths = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith("_{}.hdf5".format(dataname)):
                demo_file_paths.append(os.path.join(root, name))
                
    print('found demo data:')
    for demo_file_path in demo_file_paths:
        print('\t'+demo_file_path)

    total =0
    for (demo_idx, demo_file_path) in enumerate(demo_file_paths):
        demo_file = h5py.File(demo_file_path, 'r')
        print('processing {}...'.format(demo_file_path))

        done = np.zeros(demo_file['subtask'].shape[0], dtype='uint64')
        obs_data = {key: demo_file['obs'][key] for key in demo_file['obs'].keys()}

        ep_grp = out_data.create_group(f"demo_{demo_idx}")
        obs_grp = ep_grp.create_group(f"obs")

        for label, data in obs_data.items():
            if label not in ['right_rgb', 'left_rgb']:
                obs_grp.create_dataset(label, data=data, compression="gzip", chunks=True, dtype='f')
            else:
                obs_grp.create_dataset(label, data=data, compression="gzip", chunks=True, dtype='uint8')

        actions = np.array(demo_file['actions'])

        ep_grp.create_dataset("actions", data=actions)
        ep_grp.create_dataset("dones", data=done, dtype='uint64')
        ep_grp.create_dataset("rewards", data=done)
        ep_grp.attrs["num_samples"] = int(done.shape[0])
        ep_grp.attrs["tag"] = demo_file_path.split('/')[-2]
        total += int(done.shape[0])

        demo_file.close()

    out_file.attrs["total"] = total
    metadata = ""
    out_file.attrs["env_args"] = metadata
    out_file.close()


def merge_data(path, env_type="door", subtask=0):
    
    out_file = h5py.File(os.path.join(path, '{}_{}_merged.hdf5').format(env_type, subtask), 'w')
    out_data = out_file.create_group("data")

    total = 0
    num_eps = 0

    dataset_paths = []
    
    print('found demo data:')
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith("_{}_{}.hdf5".format(env_type, subtask)):
                dataset_paths.append(os.path.join(root, name))
                print('\t'+os.path.join(root, name))
        
    for dataset_path in dataset_paths:

        print('processing {}...'.format(dataset_path))
        dataset = h5py.File(dataset_path, 'r')

        for demo in dataset['data']:
            total += dataset['data/{}'.format(demo)].attrs['num_samples']
            ep_grp = out_data.create_group("demo_{}".format(num_eps))
            ep_grp.attrs["num_samples"] = dataset['data/{}'.format(demo)].attrs['num_samples']
            for key in dataset['data/{}'.format(demo)]:
                dataset['data/{}'.format(demo)].copy(key, ep_grp)
            num_eps += 1
            
        dataset.close()

    out_file.attrs["total"] = total
    metadata = ""
    out_file.attrs["env_args"] = metadata
    out_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='default',
                        help="")
    parser.add_argument("--mode", type=str, default='process',
                        help="")
    parser.add_argument("--env", type=str, default='door',
                        help="")
    parser.add_argument("--subtask", type=int, default=0,
                        help="")
    parser.add_argument("--dataname", type=str, default='data',
                        help="")
    args = parser.parse_args()

    path = os.path.join('./datasets', args.path)
    mode = args.mode
    subtask = args.subtask
    env_type = args.env
    dataname = args.dataname

    if mode == 'process':
        post_process(path, env_type, dataname)
    elif mode == 'subtask':
        batch_subtask_data(path, env_type, subtask, dataname)
    elif mode == 'merge':
        merge_data(path, env_type, subtask)
