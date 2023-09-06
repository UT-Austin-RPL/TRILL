from collections import OrderedDict
from pnc.draco_pnc.interface import DracoManipulationInterface
import numpy as np
import copy
from util import geom


class WheeledArmController(object):

    def __init__(self) -> None:

        self._robot_target = {'joint_pos': OrderedDict(),
                              'joint_vel': OrderedDict(),
                              'joint_trq': OrderedDict(),
                              'body_pos': OrderedDict(),
                              'body_vel': OrderedDict()}
        self._left_gripper_target = {'joint_pos': OrderedDict(),
                                     'joint_vel': OrderedDict(),
                                     'joint_trq': OrderedDict()}
        self._right_gripper_target = {'joint_pos': OrderedDict(),
                                      'joint_vel': OrderedDict(),
                                      'joint_trq': OrderedDict()}                           
        self._hand_target = {'left': OrderedDict(),
                             'right': OrderedDict()}
        self._sensor_data = OrderedDict()
        self._init_state = OrderedDict()
        # self._init_state.update(config['Simulation']['Initial State'])
        self._init_state = {
            'Joint Pos': {'base_wheel_{}'.format(key): 0.0 for key in ['fl', 'fr', 'bl', 'br']},
            'Body Pos': [0.0, 0.0, 0.0],
            'Body Quat': [0.0, 0.0, 0.0, 1.0]
        }
        self._init_state['Joint Pos'].update({'left_joint_{}'.format(idx): value 
                    for idx, value in enumerate([-0.5*np.pi, -0.5*np.pi, 0.5*np.pi, 0.5*np.pi, 0.0, 1.0*np.pi, 0.0])})
        self._init_state['Joint Pos'].update({'right_joint_{}'.format(idx): value 
                    for idx, value in enumerate([0.5*np.pi, -0.5*np.pi, -0.5*np.pi, 0.5*np.pi, 0.0, 1.0*np.pi, 0.0])})
        self.reset()


    def reset(self):

        self._robot_target['joint_pos'].update(self._init_state['Joint Pos'])
        self._robot_target['joint_vel'].update({key: 0.0 for key in self._robot_target['joint_pos'].keys()})
        self._robot_target['joint_trq'].update({key: 0.0 for key in self._robot_target['joint_pos'].keys()})
        self._robot_target['body_pos'].update({'pos': np.array(self._init_state['Body Pos']), 
                                               'quat': np.array(self._init_state['Body Quat'])})
        self._robot_target['body_vel'].update({'pos': np.zeros(3), 'rpy': np.zeros(3)})

        self._right_gripper_target['joint_pos'].update({'gripper': 0.0})
        self._right_gripper_target['joint_vel'].update({key: 0.0 for key in self._right_gripper_target['joint_pos'].keys()})
        self._right_gripper_target['joint_trq'].update({key: 0.0 for key in self._right_gripper_target['joint_pos'].keys()})

        self._left_gripper_target['joint_pos'].update({'gripper': 0.0})
        self._left_gripper_target['joint_vel'].update({key: 0.0 for key in self._left_gripper_target['joint_pos'].keys()})
        self._left_gripper_target['joint_trq'].update({key: 0.0 for key in self._left_gripper_target['joint_pos'].keys()})

        self._hand_target['right'].update({'pos': np.array([0.3, -0.3, 0.3]), 
                                           'quat': np.array([0.0, 0.70710678, 0.0, -0.70710678])})
        self._hand_target['left'].update({'pos': np.array([0.3, 0.3, 0.3]), 
                                          'quat': np.array([0.0, 0.70710678, 0.0, -0.70710678])})


    def update_sensor_data(self, sensor_data):
        pass

    def get_control(self):
        pass

    def standby(self):
        pass


class DracoController(object):

    def __init__(self, config, path_to_robot_model) -> None:
        
        # Construct Interface
        self._interface = DracoManipulationInterface(path_to_robot_model, config)
        self._robot_target = {'joint_pos': OrderedDict(),
                              'joint_vel': OrderedDict(),
                              'joint_trq': OrderedDict(),
                              'body_pos': OrderedDict(),
                              'body_vel': OrderedDict()}
        self._aux_target = {'joint_pos': OrderedDict(),
                            'joint_vel': OrderedDict(),
                            'joint_trq': OrderedDict()}
        self._left_gripper_target = {'joint_pos': OrderedDict(),
                                     'joint_vel': OrderedDict(),
                                     'joint_trq': OrderedDict()}
        self._right_gripper_target = {'joint_pos': OrderedDict(),
                                      'joint_vel': OrderedDict(),
                                      'joint_trq': OrderedDict()}
        self._hand_target = OrderedDict()
        self._sensor_data = OrderedDict()
        self._init_state = OrderedDict()
        self._init_action = OrderedDict()
        self._gripper_config = OrderedDict()
        self._aux_config = OrderedDict()

        self._init_state.update(config['Simulation']['Initial State'])
        self._init_action.update(config['Action']['Default'])
        self._gripper_config.update(config['Gripper Control'])
        self._aux_config.update(config['Aux Control'])

        self.reset()


    def reset(self, initial_pos=None):

        self._robot_target['joint_pos'].update(self._init_state['Joint Pos'])
        self._robot_target['joint_vel'].update({key: 0.0 for key in self._robot_target['joint_pos'].keys()})
        self._robot_target['joint_trq'].update({key: 0.0 for key in self._robot_target['joint_pos'].keys()})
    
        if initial_pos is not None:
            if 'pos' in initial_pos.keys():
                self._robot_target['body_pos'].update({'pos': np.array(initial_pos['pos'])})
            if 'yaw' in initial_pos.keys():
                quat = geom.euler_to_quat([0, 0, initial_pos['yaw']])
                self._robot_target['body_pos'].update({'quat': quat})
        else:
            self._robot_target['body_pos'].update({'pos': np.array(self._init_state['Body Pos']), 
                                               'quat': np.array(self._init_state['Body Quat'])})
        self._robot_target['body_vel'].update({'pos': np.zeros(3), 'rpy': np.zeros(3)})

        self._aux_target['joint_pos'].update(self._init_state['Aux Pos'])
        self._aux_target['joint_vel'].update({key: 0.0 for key in self._aux_target['joint_pos'].keys()})
        self._aux_target['joint_trq'].update({key: 0.0 for key in self._aux_target['joint_pos'].keys()})

        self._right_gripper_target['joint_pos'].update({'gripper': 0.0})
        self._right_gripper_target['joint_vel'].update({key: 0.0 for key in self._right_gripper_target['joint_pos'].keys()})
        self._right_gripper_target['joint_trq'].update({key: 0.0 for key in self._right_gripper_target['joint_pos'].keys()})

        self._left_gripper_target['joint_pos'].update({'gripper': 0.0})
        self._left_gripper_target['joint_vel'].update({key: 0.0 for key in self._left_gripper_target['joint_pos'].keys()})
        self._left_gripper_target['joint_trq'].update({key: 0.0 for key in self._left_gripper_target['joint_pos'].keys()})

        self._hand_target.update({'right_pos': np.copy(self._init_action['trajectory']['right_pos']), 
                                  'right_quat': np.copy(self._init_action['trajectory']['right_quat']),
                                  'left_pos': np.copy(self._init_action['trajectory']['left_pos']),
                                  'left_quat':  np.copy(self._init_action['trajectory']['left_quat'])})

    
    def update_sensor_data(self, sensor_data):

        self._sensor_data.update(copy.deepcopy(sensor_data))


    def update_trajectory(self, tajectory, locomotion=None):

        if not self._interface.interrupt_logic.b_walk_in_progress:
            # if state == 'in_place':
            #     self._interface.interrupt_logic._walk_in_place = True
            if locomotion == 'walk_forward':
                self._interface.interrupt_logic._walk_forward = True
            if locomotion == 'walk_backward':
                self._interface.interrupt_logic._walk_backward = True
            if locomotion == 'sidewalk_left':
                self._interface.interrupt_logic._strafe_left = True
            if locomotion == 'sidewalk_right':
                self._interface.interrupt_logic._strafe_right = True
            if locomotion == 'turning_left':
                self._interface.interrupt_logic._turn_left = True
            if locomotion == 'turning_right':
                self._interface.interrupt_logic._turn_right = True
            if locomotion == 'balance':
                self._interface.interrupt_logic._release = True

        self._hand_target.update({key: np.copy(value) for key, value in tajectory.items() if key in self._hand_target.keys()})

        base_com_quat = self._sensor_data['base_com_quat']
        base_com_pos = self._sensor_data['base_com_pos']
        rot_world_basejoint = geom.quat_to_rot(base_com_quat)
        calmped_local_rh_target_pos = np.clip(self._hand_target['right_pos'], 
                                               self._interface.config['Manipulation']['Workspace']['RH Min'], 
                                               self._interface.config['Manipulation']['Workspace']['RH Max'])
        calmped_local_lh_target_pos = np.clip(self._hand_target['left_pos'], 
                                               self._interface.config['Manipulation']['Workspace']['LH Min'], 
                                               self._interface.config['Manipulation']['Workspace']['LH Max'])
        rh_target_pos = np.dot(rot_world_basejoint, calmped_local_rh_target_pos) + base_com_pos
        lh_target_pos = np.dot(rot_world_basejoint, calmped_local_lh_target_pos) + base_com_pos

        rh_target_rot = rot_world_basejoint @ geom.quat_to_rot(self._hand_target['right_quat'])
        lh_target_rot = rot_world_basejoint @ geom.quat_to_rot(self._hand_target['left_quat'])
        rh_target_quat = geom.rot_to_quat(rh_target_rot)
        lh_target_quat = geom.rot_to_quat(lh_target_rot)

        self._interface.interrupt_logic.rh_target_pos = rh_target_pos
        self._interface.interrupt_logic.rh_target_quat = rh_target_quat
        self._interface.interrupt_logic.lh_target_pos = lh_target_pos
        self._interface.interrupt_logic.lh_target_quat = lh_target_quat


    def update_gripper_target(self, state):

        for key, value in state.items():
            target_pos = self._gripper_config['Action'][key][value]
            if key == 'left':
                self._left_gripper_target['joint_pos'].update({'gripper': target_pos})
            if key == 'right':
                self._right_gripper_target['joint_pos'].update({'gripper': target_pos})            


    def update_aux_target(self, state):

        for key, value in state.items():
            dict_target_pos = self._aux_config['Action'][key][value]
            self._aux_target['joint_pos'].update(dict_target_pos)


    def get_gripper_control(self):
        target = OrderedDict()
        target.update({'left': copy.deepcopy(self._left_gripper_target)})
        target.update({'right': copy.deepcopy(self._right_gripper_target)})

        return target


    def get_aux_control(self):
        return copy.deepcopy(self._aux_target)


    def get_control(self):

        command = self._interface.get_command(copy.deepcopy(self._sensor_data))

        del command['joint_pos']['l_knee_fe_jp']
        del command['joint_pos']['r_knee_fe_jp']
        del command['joint_vel']['l_knee_fe_jp']
        del command['joint_vel']['r_knee_fe_jp']
        del command['joint_trq']['l_knee_fe_jp']
        del command['joint_trq']['r_knee_fe_jp']

        return command


    def get_state(self):
        return self._interface._control_architecture.state


    def standby(self):

        self._interface.interrupt_logic._standby = True


    @property
    def aux_config(self):
        return self._aux_config
    
    @property
    def gripper_config(self):
        return self._gripper_config
