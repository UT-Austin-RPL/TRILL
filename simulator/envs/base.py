import os
import sys
import yaml
import time
import copy
import numpy as np

cwd = os.getcwd()
sys.path.append(cwd)

from simulator.robots import Draco3
from simulator.grippers import SakeEZGripper
from simulator.controllers import DracoController
import simulator.sim_util as sim_util

from robosuite.models.base import MujocoXML
from robosuite.utils.binding_utils import MjSim

PATH_TO_ROBOT_MODEL = os.path.expanduser(cwd+'/models/robots/draco3')
PATH_TO_WORLD_XML = os.path.expanduser(cwd+'/models/base.xml')

SIM_TIME = 0.002
TELEOP_TIME = 0.05
WBC_TIME = 0.01
RENDER_TIME = 0.033
INIT_TIME = 10 * SIM_TIME

class BaseEnv():

    def __init__(self) -> None:

        self._load_model()

        model = self.world.get_model(mode="mujoco")
        self.sim = MjSim(model)

        # Renderer
        self.renderer = None
        self.recorder = None

        # Controller
        with open(cwd+'/configs/default.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        self.action_map = self.config['Action']['Map']

        self.delay = None

    def reset(self, subtask=0, initial_pos=None, initial_qpos=None, **kwargs):

        self.sim.reset()
        self._reset_robot(initial_pos)
        self._reset_objects()
        self._reset_recorder()
        if initial_qpos is not None:
            self._reset_initial_qpos(initial_qpos)

        self.sim.forward()

        self._cur_sim_time = 0.0
        self._cur_wbc_time = 0.0
        self._cur_render_time = 0.0
        self._cur_teleop_time = 0.0

        self._subtask = subtask

        self._cur_action = copy.deepcopy(self.config['Action']['Default'])
        self._cur_obs = {'img': np.ones((10, 20), dtype=np.uint8), 'state': self.controller.get_state()} # THIS IS A DUMMY OBS

        while self._cur_sim_time < INIT_TIME:
            self._init_control()
            self.sim.step()
            self._cur_sim_time += SIM_TIME
            # self._render()

        self._update_obs()

        if self.delay is not None:
            self._action_buffer = []
            for _ in range(self.delay):
                self._action_buffer.append(copy.deepcopy(self.config['Action']['Default']))

        return self._get_obs()


    def step(self, action):

        if ('subtask' in action.keys()) and (action['subtask']):
            self._subtask += 1

        if self.delay is not None:
            self._cur_action['locomotion'] = self._action_buffer[0]['locomotion']
            self._cur_action['trajectory'].update(self._action_buffer[0]['trajectory'])
            self._cur_action['gripper'].update(self._action_buffer[0]['gripper'])
            self._cur_action['aux'].update(self._action_buffer[0]['aux'])
            self._action_buffer = self._action_buffer[1:]
            self._action_buffer.append(action)
        else:
            self._cur_action['locomotion'] = action['locomotion']
            self._cur_action['trajectory'].update(action['trajectory'])
            self._cur_action['gripper'].update(action['gripper'])
            self._cur_action['aux'].update(action['aux'])

        prv_obs = self._get_obs()
        cur_cmd = self._get_cmd()
        cur_action = self._get_action()
        self._record(time_stamp=self._cur_teleop_time, label='demo', data={'action': cur_action, 'observation': prv_obs})

        self.controller.update_trajectory(cur_cmd['trajectory'], cur_cmd['locomotion'])
        self.controller.update_gripper_target(cur_cmd['gripper'])
        self.controller.update_aux_target(cur_cmd['aux'])

        while self._cur_sim_time - self._cur_teleop_time < TELEOP_TIME:

            self._apply_control()
            self.sim.step()
            self._record(time_stamp=self._cur_sim_time)

            if self._cur_sim_time - self._cur_render_time >= RENDER_TIME:
                #self._render()
                self._cur_render_time += RENDER_TIME

            self._cur_sim_time += SIM_TIME
            
        self._cur_teleop_time += TELEOP_TIME
        self._render()
        self._update_obs()

        return self._get_obs()


    @property
    def cur_time(self):
        return self._cur_sim_time


    @property
    def done(self):
        return False


    @property
    def subtask(self):
        return self._subtask


    def render(self):
        return self._render()


    def set_renderer(self, renderer):
        self.renderer = renderer


    def set_recorder(self, recorder):
        self.recorder = recorder


    def _update_obs(self):
        joint_data = sim_util.get_joint_state(self.sim, self.robot)
        self._cur_obs['subtask'] = self._subtask
        self._cur_obs['state'] = self.controller.get_state()
        self._cur_obs['trajectory'] = sim_util.get_trajectory(self.sim, self.robot)
        self._cur_obs['joint_pos'] = joint_data['joint_pos']
        self._cur_obs['joint_vel'] = joint_data['joint_vel']

    def _get_obs(self):
        return self._cur_obs

    def _get_action(self):
        return self._cur_action

    def _get_cmd(self):
        cur_cmd = {}
        cur_cmd['trajectory'] = self._cur_action['trajectory']
        cur_cmd['locomotion'] = self.action_map['locomotion'][self._cur_action['locomotion']]
        cur_cmd['gripper'] = {key: self.action_map['gripper'][key][int(value)] for key, value in self._cur_action['gripper'].items()}
        cur_cmd['aux'] = {key: self.action_map['aux'][key][int(value)] for key, value in self._cur_action['aux'].items()}

        return cur_cmd


    def _load_model(self):
        self.world = MujocoXML(PATH_TO_WORLD_XML)
        self.robot = Draco3()
        self.grippers={}
        for key in [ 'right', 'left']:
            self.grippers[key] = SakeEZGripper(idn=key)
            self.robot.add_gripper(self.grippers[key], arm_name=self.robot.naming_prefix+self.robot._eef_name[key])
        self.world.merge(self.robot)


    def _reset_robot(self, initial_pos=None):
        self.controller = DracoController(self.config, PATH_TO_ROBOT_MODEL)
        self.controller.reset(initial_pos=initial_pos)
        sim_util.set_body_pos_vel(self.sim, self.robot, self.controller._robot_target)
        sim_util.set_motor_pos_vel(self.sim, self.robot, self.controller._robot_target)
        sim_util.set_motor_trq(self.sim, self.robot, self.controller._robot_target)
        sim_util.set_motor_impedance(self.sim, self.robot, self.controller._aux_target, **self.controller.aux_config['Gain'])
        sim_util.set_motor_impedance(self.sim, self.grippers['left'], self.controller._left_gripper_target, **self.controller.gripper_config['Gain']['left'])
        sim_util.set_motor_impedance(self.sim, self.grippers['right'], self.controller._right_gripper_target, **self.controller.gripper_config['Gain']['right'])


    def _reset_objects(self):
        pass
        

    def _render(self):
        if self.renderer == None:
            return
        else:
            return self.renderer.render()


    def _record(self, **kwargs):
        if self.recorder == None:
            return
        else:
            return self.recorder.record(**kwargs)

    def _reset_recorder(self, **kwargs):
        if self.recorder == None:
            return
        else:
            return self.recorder.reset(**kwargs)

    def _reset_initial_qpos(self, initial_qpos):
        self.sim.data.qpos[:] = np.copy(initial_qpos)


    def _init_control(self):

        self.controller.standby()
        sensor_data = sim_util.get_sensor_data(self.sim, self.robot)
        self.controller.update_sensor_data(sensor_data)
        sim_util.set_motor_impedance(self.sim, self.robot, self.controller._robot_target, 500.0, 10)


    def _apply_control(self):

        if self._cur_sim_time - self._cur_wbc_time >= WBC_TIME:
            
            sensor_data = sim_util.get_sensor_data(self.sim, self.robot)
            self.controller.update_sensor_data(sensor_data)

            control = self.controller.get_control()
            sim_util.set_motor_trq(self.sim, self.robot, control)
            self._record(time_stamp=self._cur_wbc_time, label='wbc', data={'control': control, 'sensor': sensor_data, 'state': self.controller.get_state()})
            
            self._cur_wbc_time += WBC_TIME

        aux_control = self.controller.get_aux_control()
        aux_gain = self.controller.aux_config['Gain']
        sim_util.set_motor_impedance(self.sim, self.robot, aux_control, **aux_gain)

        gripper_control = self.controller.get_gripper_control()
        gripper_gain = self.controller.gripper_config['Gain']
        for arm in ['left', 'right']:
            sim_util.set_motor_impedance(self.sim, self.grippers[arm], gripper_control[arm], **gripper_gain[arm])
            
    def _get_robot_pos(self):
        return np.copy(self.controller._sensor_data['base_com_pos'])

    def _get_robot_quat(self):
        return np.copy(self.controller._sensor_data['base_com_quat'])

    def _check_success(self):
        raise NotImplementedError

    @property
    def robot_pos(self):
        return self._get_robot_pos()

    @property
    def robot_quat(self):
        return self._get_robot_quat()

    @property
    def success(self):
        self._check_success()
        return self._success
