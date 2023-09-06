from collections import OrderedDict
import numpy as np
import mujoco

from util import geom, liegroup


def get_mujoco_objects(sim):
    return sim.model._model, sim.data._data


def get_link_iso(sim, robot, link_name):
    model, data = get_mujoco_objects(sim)
    link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, link_name)
    pos = np.array(data.xpos[link_id])
    quat = np.array(data.xquat[link_id])[[1,2,3,0]]
    rot = geom.quat_to_rot(quat)
    return liegroup.RpToTrans(rot, pos)


def get_link_vel(sim, robot, link_name):
    model, data = get_mujoco_objects(sim)
    link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, link_name)
    velp = np.copy(data.cvel(link_id))[:3]
    velr = np.copy(data.cvel(link_id))[3:]
    vel = np.concatenate((velr, velp))
    return vel


def get_body_pos_vel(sim, robot):
    model, data = get_mujoco_objects(sim)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, robot.naming_prefix+'root')
    joint_qposadr = model.jnt_qposadr[joint_id]
    joint_qveladr = model.jnt_dofadr[joint_id]

    state = {'body_pos': {
                'pos': np.copy(data.qpos[joint_qposadr:joint_qposadr+3]), 
                'quat': np.copy(data.qpos[joint_qposadr+3:joint_qposadr+7])[[1,2,3,0]]
                },
             'body_vel': {
                'pos': np.copy(data.qvel[joint_qveladr:joint_qveladr+3]), 
                'rpy': np.copy(data.qvel[joint_qveladr+3:joint_qveladr+6])
                }
             }

    return state


def set_motor_impedance(sim, robot, command, kp, kd):
    model, data = get_mujoco_objects(sim)
    key_map = robot.key_map
    trq_applied = OrderedDict()
    for (pnc_key, pos_des), (_, vel_des), (_, trq_des) in zip(
            command['joint_pos'].items(), command['joint_vel'].items(),
            command['joint_trq'].items()):
        mujoco_joint_key = key_map['joint'][pnc_key]
        mujoco_actuator_key = key_map['actuator'][pnc_key]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mujoco_joint_key)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco_actuator_key)
        joint_qposadr = model.jnt_qposadr[joint_id]
        joint_qveladr = model.jnt_dofadr[joint_id]
        joint_pos = data.qpos[joint_qposadr]
        joint_vel = data.qvel[joint_qveladr]
        if type(kp) == dict: kp_val = kp[pnc_key]
        else: kp_val = kp
        if type(kd) == dict: kd_val = kd[pnc_key]
        else: kd_val = kd
        trq_applied[actuator_id] = trq_des \
                + kp_val * (pos_des - joint_pos)\
                + kd_val * (vel_des - joint_vel)
    data.ctrl[list(trq_applied.keys())] = list(trq_applied.values())


def set_motor_trq(sim, robot, command):
    model, data = get_mujoco_objects(sim)
    key_map = robot.key_map
    trq_applied = OrderedDict()
    for pnc_key, trq_des in command['joint_trq'].items():
        mujoco_actuator_key = key_map['actuator'][pnc_key]
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco_actuator_key)
        # print(actuator_id)
        trq_applied[actuator_id] = trq_des
    data.ctrl[list(trq_applied.keys())] = list(trq_applied.values())


def set_motor_pos(sim, robot, state):
    model, data = get_mujoco_objects(sim)
    key_map = robot.key_map
    pos_applied = OrderedDict()
    vel_applied = OrderedDict()
    for pnc_key, pos_des in state['joint_pos'].items():
        mujoco_joint_key = key_map['joint'][pnc_key]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mujoco_joint_key)
        joint_qposadr = model.jnt_qposadr[joint_id]
        pos_applied[joint_qposadr] = pos_des
    data.qpos[list(pos_applied.keys())] = list(pos_applied.values())

def set_ball_pos(sim, robot, state):
    model, data = get_mujoco_objects(sim)
    key_map = robot.key_map
    pos_applied = OrderedDict()
    vel_applied = OrderedDict()
    for pnc_key, pos_des in state['joint_pos'].items():
        mujoco_joint_key = key_map['joint'][pnc_key]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mujoco_joint_key)
        joint_qposadr = model.jnt_qposadr[joint_id]
        for qposadr_idx, val_idx in enumerate([3, 0, 1, 2]): pos_applied[joint_qposadr+qposadr_idx] = pos_des[val_idx]
    data.qpos[list(pos_applied.keys())] = list(pos_applied.values())



def set_motor_pos_vel(sim, robot, state):
    model, data = get_mujoco_objects(sim)
    key_map = robot.key_map
    pos_applied = OrderedDict()
    vel_applied = OrderedDict()
    for (pnc_key, pos_des), (_, vel_des) in zip(state['joint_pos'].items(), state['joint_vel'].items()):
        mujoco_joint_key = key_map['joint'][pnc_key]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mujoco_joint_key)
        joint_qposadr = model.jnt_qposadr[joint_id]
        joint_qveladr = model.jnt_dofadr[joint_id]
        pos_applied[joint_qposadr] = pos_des
        vel_applied[joint_qveladr] = vel_des
    data.qpos[list(pos_applied.keys())] = list(pos_applied.values())
    data.qvel[list(vel_applied.keys())] = list(vel_applied.values())


def set_body_pos_vel(sim, robot, state):
    model, data = get_mujoco_objects(sim)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, robot.naming_prefix+'root')
    joint_qposadr = model.jnt_qposadr[joint_id]
    joint_qveladr = model.jnt_dofadr[joint_id]
    data.qpos[joint_qposadr:joint_qposadr+7] = np.concatenate((state['body_pos']['pos'], state['body_pos']['quat'][[3, 0, 1, 2]]))
    data.qvel[joint_qveladr:joint_qveladr+6] = np.concatenate((state['body_vel']['pos'], state['body_vel']['rpy']))


def set_ball_pos_vel(sim, robot, state):
    model, data = get_mujoco_objects(sim)
    key_map = robot.key_map
    pos_applied = OrderedDict()
    vel_applied = OrderedDict()
    for (pnc_key, pos_des), (_, vel_des) in zip(state['joint_pos'].items(), state['joint_vel'].items()):
        mujoco_joint_key = key_map['joint'][pnc_key]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mujoco_joint_key)
        joint_qposadr = model.jnt_qposadr[joint_id]
        joint_qveladr = model.jnt_dofadr[joint_id]
        for qposadr_idx, val_idx in enumerate([3, 0, 1, 2]): pos_applied[joint_qposadr+qposadr_idx] = pos_des[val_idx]
        for idx in range(3): vel_applied[joint_qveladr+idx] = vel_des[idx]
    data.qpos[list(pos_applied.keys())] = list(pos_applied.values())
    data.qvel[list(vel_applied.keys())] = list(vel_applied.values())


def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def get_sensor_data(sim, robot):
    """
    Parameters
    ----------
    joint_id (dict):
        Joint ID Dict
    link_id (dict):
        Link ID Dict
    pos_basejoint_to_basecom (np.ndarray):
        3d vector from base joint frame to base com frame
    rot_basejoint_to_basecom (np.ndarray):
        SO(3) from base joint frame to base com frame
    b_fixed_Base (bool);
        Whether the robot is floating or fixed
    Returns
    -------
    sensor_data (dict):
        base_com_pos (np.array):
            base com pos in world
        base_com_quat (np.array):
            base com quat in world
        base_com_lin_vel (np.array):
            base com lin vel in world
        base_com_ang_vel (np.array):
            base com ang vel in world
        base_joint_pos (np.array):
            base pos in world
        base_joint_quat (np.array):
            base quat in world
        base_joint_lin_vel (np.array):
            base lin vel in world
        base_joint_ang_vel (np.array):
            base ang vel in world
        joint_pos (dict):
            Joint pos
        joint_vel (dict):
            Joint vel
        b_rf_contact (bool):
            Right Foot Contact Switch
        b_lf_contact (bool):
            Left Foot Contact Switch
    """
    model, data = get_mujoco_objects(sim)
    sensor_data = OrderedDict()

    base_com_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, robot.naming_prefix+'base_com')
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, robot.naming_prefix+'base')
    lh_eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, robot.naming_prefix+'left_hand')
    rh_eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, robot.naming_prefix+'right_hand')

    base_com_jacp = np.zeros((3, model.nv))
    base_com_jacr = np.zeros((3, model.nv))
    base_jacp = np.zeros((3, model.nv))
    base_jacr = np.zeros((3, model.nv))

    mujoco.mj_jacBody(model, data, base_com_jacp, base_com_jacr, base_com_id)
    mujoco.mj_jacBody(model, data, base_jacp, base_jacr, base_id)

    base_com_pos = np.copy(data.xpos[base_com_id])
    base_com_quat = np.copy(data.xquat[base_com_id][[1,2,3,0]])
    sensor_data['base_com_pos'] = base_com_pos
    sensor_data['base_com_quat'] = base_com_quat

    base_com_lin_vel = np.dot(base_com_jacp, data.qvel) # np.copy(data.cvel[base_com_id])[:3]
    base_com_ang_vel = np.dot(base_com_jacr, data.qvel) # np.copy(data.cvel[base_com_id])[3:]
    sensor_data['base_com_lin_vel'] = np.asarray(base_com_lin_vel)
    sensor_data['base_com_ang_vel'] = np.asarray(base_com_ang_vel)

    base_joint_pos = np.copy(data.xpos[base_id])
    base_joint_quat = np.copy(data.xquat[base_id][[1,2,3,0]])
    sensor_data['base_joint_pos'] = base_joint_pos
    sensor_data['base_joint_quat'] = base_joint_quat

    base_joint_lin_vel = np.dot(base_jacp, data.qvel) # np.copy(data.cvel[base_id])[:3]
    base_joint_ang_vel = np.dot(base_jacr, data.qvel) # np.copy(data.cvel[base_id])[3:]
    sensor_data['base_joint_lin_vel'] = base_joint_lin_vel
    sensor_data['base_joint_ang_vel'] = base_joint_ang_vel

    rf_height = get_link_iso(sim, robot, 'link_left_foot')[2, 3]
    lf_height = get_link_iso(sim, robot, 'link_right_foot')[2, 3]

    sensor_data['b_rf_contact'] = True if rf_height <= 0.01 else False
    sensor_data['b_lf_contact'] = True if lf_height <= 0.01 else False

    rot_world_basejoint = geom.quat_to_rot(base_com_quat)
    lh_eef_pos = np.dot(rot_world_basejoint.transpose(), data.xpos[lh_eef_id]-base_com_pos)
    rh_eef_pos = np.dot(rot_world_basejoint.transpose(), data.xpos[rh_eef_id]-base_com_pos)
    sensor_data['lh_eef_pos'] = lh_eef_pos
    sensor_data['rh_eef_pos'] = rh_eef_pos

    # Joint Quantities
    joint_data = get_joint_state(sim, robot)
    sensor_data.update(joint_data)

    return sensor_data


def get_joint_state(sim, robot):
    
    model, data = get_mujoco_objects(sim)
    key_map = robot.key_map
    joint_data = OrderedDict()

    joint_data['joint_pos'] = OrderedDict()
    joint_data['joint_vel'] = OrderedDict()
    for pnc_key in key_map['joint'].keys():
        mujoco_key = key_map['joint'][pnc_key]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mujoco_key)
        joint_qposadr = model.jnt_qposadr[joint_id]
        joint_qveladr = model.jnt_dofadr[joint_id]
        joint_data['joint_pos'][pnc_key] = data.qpos[joint_qposadr]
        joint_data['joint_vel'][pnc_key] = data.qvel[joint_qveladr]

    return joint_data


def get_trajectory(sim, robot):

    model, data = get_mujoco_objects(sim)
    trajectory_data = OrderedDict()

    base_com_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, robot.naming_prefix+'base_com')
    lh_eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, robot.naming_prefix+'left_hand')
    rh_eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, robot.naming_prefix+'right_hand')
    lf_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, robot.naming_prefix+'left_foot')
    rf_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, robot.naming_prefix+'right_foot')

    base_com_pos = np.copy(data.xpos[base_com_id])
    base_com_quat = np.copy(data.xquat[base_com_id][[1,2,3,0]])

    rot_world_basejoint = geom.quat_to_rot(base_com_quat)
    lh_eef_pos = np.dot(rot_world_basejoint.transpose(), data.xpos[lh_eef_id]-base_com_pos)
    rh_eef_pos = np.dot(rot_world_basejoint.transpose(), data.xpos[rh_eef_id]-base_com_pos)
    lf_foot_pos = np.dot(rot_world_basejoint.transpose(), data.xpos[lf_foot_id]-base_com_pos)
    rf_foot_pos = np.dot(rot_world_basejoint.transpose(), data.xpos[rf_foot_id]-base_com_pos)

    lh_eef_rot = np.dot(rot_world_basejoint.transpose(), geom.quat_to_rot(data.xquat[lh_eef_id][[1,2,3,0]]))
    rh_eef_rot = np.dot(rot_world_basejoint.transpose(), geom.quat_to_rot(data.xquat[rh_eef_id][[1,2,3,0]]))
    lf_foot_rot = np.dot(rot_world_basejoint.transpose(), geom.quat_to_rot(data.xquat[lf_foot_id][[1,2,3,0]]))
    rf_foot_rot = np.dot(rot_world_basejoint.transpose(), geom.quat_to_rot(data.xquat[rf_foot_id][[1,2,3,0]]))

    lh_eef_quat = geom.rot_to_quat(lh_eef_rot)
    rh_eef_quat = geom.rot_to_quat(rh_eef_rot)
    lf_foot_quat = geom.rot_to_quat(lf_foot_rot)
    rf_foot_quat = geom.rot_to_quat(rf_foot_rot)

    trajectory_data['lh_eef_pos'] = lh_eef_pos
    trajectory_data['rh_eef_pos'] = rh_eef_pos
    trajectory_data['lf_foot_pos'] = lf_foot_pos
    trajectory_data['rf_foot_pos'] = rf_foot_pos

    trajectory_data['lh_eef_quat'] = lh_eef_quat
    trajectory_data['rh_eef_quat'] = rh_eef_quat
    trajectory_data['lf_foot_quat'] = lf_foot_quat
    trajectory_data['rf_foot_quat'] = rf_foot_quat

    return trajectory_data


def get_global_trajectory(sim, robot):

    model, data = get_mujoco_objects(sim)
    trajectory_data = OrderedDict()

    lh_eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, robot.naming_prefix+'left_hand')
    rh_eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, robot.naming_prefix+'right_hand')
    lf_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, robot.naming_prefix+'left_foot')
    rf_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, robot.naming_prefix+'right_foot')

    lh_eef_pos = np.copy(data.xpos[lh_eef_id])
    rh_eef_pos = np.copy(data.xpos[rh_eef_id])
    lf_foot_pos = np.copy(data.xpos[lf_foot_id])
    rf_foot_pos = np.copy(data.xpos[rf_foot_id])

    lh_eef_quat = np.copy(data.xquat[lh_eef_id][[1,2,3,0]])
    rh_eef_quat = np.copy(data.xquat[rh_eef_id][[1,2,3,0]])
    lf_foot_quat = np.copy(data.xquat[lf_foot_id][[1,2,3,0]])
    rf_foot_quat = np.copy(data.xquat[rf_foot_id][[1,2,3,0]])

    trajectory_data['lh_eef_pos'] = lh_eef_pos
    trajectory_data['rh_eef_pos'] = rh_eef_pos
    trajectory_data['lf_foot_pos'] = lf_foot_pos
    trajectory_data['rf_foot_pos'] = rf_foot_pos

    trajectory_data['lh_eef_quat'] = lh_eef_quat
    trajectory_data['rh_eef_quat'] = rh_eef_quat
    trajectory_data['lf_foot_quat'] = lf_foot_quat
    trajectory_data['rf_foot_quat'] = rf_foot_quat

    return trajectory_data


def transform_local_trajectory(sim, robot, global_trajectory):
    
    model, data = get_mujoco_objects(sim)
    trajectory_data = OrderedDict()

    base_com_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_XBODY, robot.naming_prefix+'base_com')

    base_com_pos = np.copy(data.xpos[base_com_id])
    base_com_quat = np.copy(data.xquat[base_com_id][[1,2,3,0]])

    rot_world_basejoint = geom.quat_to_rot(base_com_quat)
    lh_eef_pos = np.dot(rot_world_basejoint.transpose(), global_trajectory['lh_eef_pos']-base_com_pos)
    rh_eef_pos = np.dot(rot_world_basejoint.transpose(), global_trajectory['rh_eef_pos']-base_com_pos)
    lf_foot_pos = np.dot(rot_world_basejoint.transpose(), global_trajectory['lf_foot_pos']-base_com_pos)
    rf_foot_pos = np.dot(rot_world_basejoint.transpose(), global_trajectory['rf_foot_pos']-base_com_pos)

    lh_eef_rot = np.dot(rot_world_basejoint.transpose(), geom.quat_to_rot(global_trajectory['lh_eef_quat']))
    rh_eef_rot = np.dot(rot_world_basejoint.transpose(), geom.quat_to_rot(global_trajectory['rh_eef_quat']))
    lf_foot_rot = np.dot(rot_world_basejoint.transpose(), geom.quat_to_rot(global_trajectory['lf_foot_quat']))
    rf_foot_rot = np.dot(rot_world_basejoint.transpose(), geom.quat_to_rot(global_trajectory['rf_foot_quat']))

    lh_eef_quat = geom.rot_to_quat(lh_eef_rot)
    rh_eef_quat = geom.rot_to_quat(rh_eef_rot)
    lf_foot_quat = geom.rot_to_quat(lf_foot_rot)
    rf_foot_quat = geom.rot_to_quat(rf_foot_rot)

    trajectory_data['lh_eef_pos'] = lh_eef_pos
    trajectory_data['rh_eef_pos'] = rh_eef_pos
    trajectory_data['lf_foot_pos'] = lf_foot_pos
    trajectory_data['rf_foot_pos'] = rf_foot_pos

    trajectory_data['lh_eef_quat'] = lh_eef_quat
    trajectory_data['rh_eef_quat'] = rh_eef_quat
    trajectory_data['lf_foot_quat'] = lf_foot_quat
    trajectory_data['rf_foot_quat'] = rf_foot_quat

    return trajectory_data

