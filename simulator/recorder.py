import numpy as np
import json 
import h5py
import yaml
import os
import copy
from .envs import SIM_TIME, RENDER_TIME, INIT_TIME, WBC_TIME, TELEOP_TIME
from tensorboardX import SummaryWriter
import collections

class HDF5Recorder():
    def __init__(self, config, sim, file_path='./data') -> None:

        self._config = config
        self._sim = sim
        self._path = file_path

        self._sim_buffer = None
        self._wbc_buffer = None
        self._demo_buffer = None


    def reset(self):

        self._sim_buffer = None
        self._wbc_buffer = None
        self._demo_buffer = None


    def record(self, time_stamp, label="sim", data=None):

        if label == "sim":
            if self._sim_buffer is None:
                self._sim_buffer = {'time': collections.deque(),
                                    'qpos': collections.deque(),
                                    'qvel': collections.deque(),
                                    'ctrl': collections.deque(),
                                    }

            self._sim_buffer["time"].append(time_stamp)
            self._sim_buffer["qpos"].append(np.copy(self._sim.data.qpos))
            self._sim_buffer["qvel"].append(np.copy(self._sim.data.qvel))
            self._sim_buffer["ctrl"].append(np.copy(self._sim.data.ctrl))

        elif label == "demo":
            
            if self._demo_buffer is None:
                self._demo_buffer = {'time': collections.deque()}
                self._demo_buffer.update({'action/{}/{}'.format(key,subkey): collections.deque()
                                            for key in data['action'].keys() if key not in ['locomotion']
                                            for subkey in data['action'][key].keys()
                                            })
                self._demo_buffer.update({'observation/{}/{}'.format(key,subkey): collections.deque()
                                            for key in data['observation'].keys() if key not in ['state', 'img', 'subtask']
                                            for subkey in data['observation'][key].keys()
                                            })
                self._demo_buffer.update({'observation/subtask': collections.deque(),
                                          'observation/state': collections.deque(),
                                          'observation/img': collections.deque(),
                                          'action/locomotion': collections.deque() })

            self._demo_buffer["time"].append(time_stamp)

            for key in data['observation'].keys():
                if key in ['state', 'img', 'subtask']:
                    self._demo_buffer["observation/"+key].append(data['observation'][key])
                else:
                    for subkey, value in data['observation'][key].items():
                        self._demo_buffer["observation/{}/{}".format(key,subkey)].append(value)

            for key in data['action'].keys():
                if key in ['locomotion']:
                    self._demo_buffer["action/"+key].append(data['action'][key])
                else:
                    for subkey, value in data['action'][key].items():
                        self._demo_buffer["action/{}/{}".format(key,subkey)].append(value)

        elif label == "wbc":

            if self._wbc_buffer is None:
                self._wbc_buffer = {'time': collections.deque(), 
                                    'state': collections.deque()} 
                self._wbc_buffer.update({"control/{}/{}".format(key,subkey): collections.deque() 
                                            for key in data['control'].keys() if key in ['joint_pos', 'joint_vel', 'joint_trq']
                                            for subkey in data['control'][key].keys()
                                            })
                self._wbc_buffer.update({"control/{}".format(key): collections.deque() 
                                            for key in data['control'].keys() if key not in ['joint_pos', 'joint_vel', 'joint_trq']
                                            })
                self._wbc_buffer.update({"sensor/{}/{}".format(key,subkey): collections.deque() 
                                            for key in data['sensor'].keys() if key in ['joint_pos', 'joint_vel', 'joint_trq']
                                            for subkey in data['sensor'][key].keys()
                                            })
                self._wbc_buffer.update({"sensor/{}".format(key): collections.deque() 
                                            for key in data['sensor'].keys() if key not in ['joint_pos', 'joint_vel', 'joint_trq']
                                            })

            self._wbc_buffer["time"].append(time_stamp)
            self._wbc_buffer["state"].append(data["state"])

            for key in data['control'].keys():
                if key in ['joint_pos', 'joint_vel', 'joint_trq']:
                    for subkey, value in data['control'][key].items():
                        self._wbc_buffer["control/{}/{}".format(key,subkey)].append(value)
                else:
                    self._wbc_buffer["control/{}".format(key)].append(data['control'][key])

            for key in data['sensor'].keys():
                if key in ['joint_pos', 'joint_vel', 'joint_trq']:
                    for subkey, value in data['sensor'][key].items():
                        self._wbc_buffer["sensor/{}/{}".format(key,subkey)].append(value)
                else:
                    self._wbc_buffer['sensor/{}'.format(key)].append(data['sensor'][key])


    def close(self):

        os.makedirs(self._path, exist_ok=True)

        self._config_file = open(os.path.join(self._path, 'config.yaml'), 'wb')
        self._config_file.write(yaml.dump(self._config).encode())
        self._config_file.close()

        self._env_file = open(os.path.join(self._path, 'env.xml'), 'wb')
        self._env_file.write(self._sim.model.get_xml().encode())
        self._env_file.close()

        self._sim_file = h5py.File(os.path.join(self._path, 'sim.hdf5'), 'w')
        for label, buffer in self._sim_buffer.items():
            self._sim_file.create_dataset(label, data=np.array(buffer), compression="gzip", chunks=True, dtype="f")
        self._sim_file.close()

        self._wbc_file = h5py.File(os.path.join(self._path, 'wbc.hdf5'), 'w')
        for label, buffer in self._wbc_buffer.items():
            self._wbc_file.create_dataset(label, data=np.array(buffer), compression="gzip", chunks=True, dtype="f")
        self._wbc_file.close()

        self._demo_file = h5py.File(os.path.join(self._path, 'demo.hdf5'), 'w')
        for label, buffer in self._demo_buffer.items():
            self._demo_file.create_dataset(label, data=np.array(buffer), compression="gzip", chunks=True, dtype="f")
        self._demo_file.close()


class Player():
    def __init__(self, env, mode, file_path='./data', post_fcn=None) -> None:

        # self.world = MujocoXML(os.path.join(file_path, 'env.xml'))
        # self.sim = MjSim(self.world.get_model(mode="mujoco"))
        self._config = yaml.load(open(os.path.join(file_path, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
        self._env = env

        self._renderer_right = None
        self._renderer_left = None

        self._post_fcn = post_fcn

        self._mode = mode
        self._load_data(file_path)


    def plot(self):
        writer = SummaryWriter()
        for sensor_line, control_line in zip(self._wbc_sensor_data, self._wbc_control_data):
            sensor_dict = json.loads(sensor_line)
            control_dict = json.loads(control_line)

            assert sensor_dict['time_stamp'] == control_dict['time_stamp']
            time_stamp = int(1000*control_dict['time_stamp'])

            for data in ['joint_pos', 'joint_vel']:
                for key in control_dict[data].keys():
                    writer.add_scalar('{}_error/{}'.format(data, key), 
                                    control_dict[data][key]-sensor_dict[data][key], 
                                    time_stamp)
                    writer.add_scalar('{}_command/{}'.format(data, key), 
                                    control_dict[data][key], 
                                    time_stamp)
                    writer.add_scalar('{}_sensor/{}'.format(data, key), 
                                    sensor_dict[data][key], 
                                    time_stamp)

            for key in control_dict['joint_trq'].keys():
                writer.add_scalar('joint_trq_command/{}'.format(key), 
                                control_dict['joint_pos'][key], 
                                time_stamp)

            reaction_force = control_dict['reaction_force']
            for idx, key in enumerate(['left_lin_x', 'left_lin_y', 'left_lin_z', 'left_ang_x', 'left_ang_y', 'left_ang_z',
                                       'right_lin_x', 'right_lin_y', 'right_lin_z', 'right_ang_x', 'right_ang_y', 'right_ang_z']):
                writer.add_scalar('reaction_force_command/{}'.format(key), reaction_force[idx], time_stamp)


    def reset(self):

        self._demo_count = 0
        self._sim_count = 0
        self._subtask = 0

        if self._mode == 'replay':

            self._env._reset_objects()
            self._env._reset_recorder()

            self._cur_sim_time = 0.0
            self._cur_render_time = 0.0
            self._cur_teleop_time = 0.0
            
            while self._cur_sim_time < INIT_TIME:
                self._env.sim.data.qpos[:] = self._read_qpos()
                self._env.sim.forward()
                self._cur_sim_time += SIM_TIME

            if self._post_fcn is not None:
                self._post_fcn(self, mode="reset")
        else:
            initial_qpos = self._read_qpos()
            self._env.reset(initial_qpos=initial_qpos)


    def step(self):

        if self._mode == 'replay':

            if self._post_fcn is not None:
                self._post_fcn(self, mode="step")

            while (self._cur_sim_time - self._cur_teleop_time < TELEOP_TIME) and not self.done:
                self._env.sim.data.qpos[:] = self._read_qpos()
                self._env.sim.forward()
                self._sim_count += 1

                if self._cur_sim_time - self._cur_render_time >= RENDER_TIME:
                    self._render()
                    self._cur_render_time += RENDER_TIME

                self._cur_sim_time += SIM_TIME

            self._cur_teleop_time += TELEOP_TIME
        else:
            action = self._read_action()
            action['subtask'] = self._subtask
            self._env.step(action)
        self._demo_count += 1


    def set_stereo_renderer(self, renderer_right, renderer_left):
        self._renderer_right = renderer_right
        self._renderer_left = renderer_left


    def set_renderer(self, renderer):
        self._env.set_renderer(renderer)

    def close(self):
        if self._post_fcn is not None:
            self._post_fcn(self)

    def _load_data(self, file_path):
        NotImplementedError

    def _read_qpos(self):
        NotImplementedError

    def _read_action(self):
        NotImplementedError

    def _read_obs(self):
        NotImplementedError

    def _render(self):
        if self._env.renderer == None:
            return
        else:
            return self._env.render()

    def _get_stereo(self):
        if self._renderer_right == None or self._renderer_left == None:
            return
        else:
            return {'right': self._renderer_right.render(), 'left': self._renderer_left.render()}
        
    @property
    def done(self):
        return self._demo_count >= self._demo_length or self._sim_count >= self._sim_length


class HDF5Player(Player):

    def _load_data(self, file_path):

        self._path = file_path

        self._sim_file = h5py.File(os.path.join(self._path, 'sim.hdf5'), 'r')
        self._qpos_data = self._sim_file['qpos'][:]
        self._qvel_data = self._sim_file['qvel'][:]
        self._sim_file.close()

        self._demo_file = h5py.File(os.path.join(self._path, 'demo.hdf5'), 'r')
        self._action_data = {}
        self._action_data['locomotion'] = np.copy(self._demo_file['action']['locomotion'])
        self._action_data.update({key: 
                                {subkey: np.copy(self._demo_file['action'][key][subkey])
                                for subkey in self._demo_file['action'][key].keys() 
                                }
                                for key in self._demo_file['action'].keys() if key not in ['locomotion']
                            })

        self._observation_data = {}
        self._observation_data['state'] = np.copy(self._demo_file['observation']['state'])
        self._observation_data['img'] = np.copy(self._demo_file['observation']['img'])
        self._observation_data['subtask'] = np.copy(self._demo_file['observation']['subtask'])
        self._observation_data.update({key: 
                                {subkey: np.copy(self._demo_file['observation'][key][subkey])
                                for subkey in self._demo_file['observation'][key].keys() 
                                }
                                for key in self._demo_file['observation'].keys() if key not in ['state', 'img', 'subtask']
                            })

        self._demo_file.close()
        self._demo_length = self._action_data['locomotion'].shape[0]
        self._sim_length = self._qpos_data.shape[0]


    def _read_qpos(self):
        qpos = np.copy(self._qpos_data[self._sim_count])
        return qpos


    def _read_action(self):

        action = {}
        action['locomotion'] = self._action_data['locomotion'][self._demo_count]
        action.update({key: 
                        {subkey: self._action_data[key][subkey][self._demo_count]
                        for subkey in self._action_data[key].keys() 
                        }
                        for key in self._action_data.keys() if key not in ['locomotion']
                       })

        return action


    def _read_obs(self):

        observation = {}
        observation['state'] = self._observation_data['state'][self._demo_count]
        observation['subtask'] = self._observation_data['subtask'][self._demo_count]
        observation.update({key: 
                                {subkey: self._observation_data[key][subkey][self._demo_count]
                                for subkey in self._observation_data[key].keys() 
                                }
                                for key in self._observation_data.keys() if key not in ['state', 'img', 'subtask']
                            })

        return observation


    def close(self):
        
        super().close()

        if self._post_fcn is not None:
            self._post_fcn(self, mode="close")


if __name__ == "__main__":
    pass
