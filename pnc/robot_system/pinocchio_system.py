import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

from collections import OrderedDict

import numpy as np
import pinocchio as pin

from .base import RobotSystem
from util import geom


class PinocchioRobotSystem(RobotSystem):
    """
    Pinnochio considers floating base with 7 positions and 6 velocities with the
    order of [x, y, z, quat_x, quat_y, quat_z, quat_w, joints] and
    [xdot, ydot, zdot, ang_x, ang_y, ang_z, joints].
    Note that first six element of generalized velocities are represented in the
    base joint frame acting on the base joint frame.
    """

    def __init__(self, urdf_file, package_dir, b_fixed_base, b_print_info=False):
        super(PinocchioRobotSystem, self).__init__(
            urdf_file, package_dir, b_fixed_base, b_print_info
        )

    def _config_robot(self, urdf_file, package_dir):
        if self._b_fixed_base:
            # Fixed based robot
            (
                self._model,
                self._collision_model,
                self._visual_model,
            ) = pin.buildModelsFromUrdf(urdf_file, package_dir)
            self._n_floating = 0
        else:
            # Floating based robot
            (
                self._model,
                self._collision_model,
                self._visual_model,
            ) = pin.buildModelsFromUrdf(
                urdf_file, package_dir, pin.JointModelFreeFlyer()
            )
            self._n_floating = 6

        self._data, self._collision_data, self._visual_data = pin.createDatas(
            self._model, self._collision_model, self._visual_model
        )

        self._n_q = self._model.nq
        self._n_q_dot = self._model.nv
        self._n_a = self._n_q_dot - self._n_floating

        passing_idx = 0
        for j_id, j_name in enumerate(self._model.names):
            if j_name == "root_joint" or j_name == "universe":
                passing_idx += 1
            else:
                self._joint_id[j_name] = j_id - passing_idx

        for f_id, frame in enumerate(self._model.frames):
            if frame.name == "root_joint" or frame.name == "universe":
                pass
            else:
                if f_id % 2 == 0:
                    # Link
                    link_id = int(f_id / 2 - 1)
                    self._link_id[frame.name] = link_id
                else:
                    # Joint
                    pass

        assert len(self._joint_id) == self._n_a

        self._total_mass = sum([inertia.mass for inertia in self._model.inertias])

        if self._b_fixed_base:
            self._joint_pos_limit = np.stack(
                [self._model.lowerPositionLimit, self._model.upperPositionLimit], axis=1
            )
        else:
            self._joint_pos_limit = np.stack(
                [self._model.lowerPositionLimit, self._model.upperPositionLimit], axis=1
            )[self._n_floating + 1 : self._n_floating + 1 + self._n_a, :]
        self._joint_vel_limit = np.stack(
            [-self._model.velocityLimit, self._model.velocityLimit], axis=1
        )[self._n_floating : self._n_floating + self._n_a, :]
        self._joint_trq_limit = np.stack(
            [-self._model.effortLimit, self._model.effortLimit], axis=1
        )[self._n_floating : self._n_floating + self._n_a, :]

    def get_q_idx(self, joint_id):
        if type(joint_id) is list:
            return [self.get_q_idx(j_id) for j_id in joint_id]
        else:
            return self._model.joints[self._model.getJointId(joint_id)].idx_q

    def get_q_dot_idx(self, joint_id):
        if type(joint_id) is list:
            return [self.get_q_dot_idx(j_id) for j_id in joint_id]
        else:
            return self._model.joints[self._model.getJointId(joint_id)].idx_v

    def get_joint_idx(self, joint_id):
        if type(joint_id) is list:
            return [self.get_joint_idx(j_id) for j_id in joint_id]
        else:
            return (
                self._model.joints[self._model.getJointId(joint_id)].idx_v
                - self._n_floating
            )

    def create_cmd_ordered_dict(self, joint_pos_cmd, joint_vel_cmd, joint_trq_cmd):
        command = OrderedDict()
        command["joint_pos"] = OrderedDict()
        command["joint_vel"] = OrderedDict()
        command["joint_trq"] = OrderedDict()

        for k, v in self._joint_id.items():
            command["joint_pos"][k] = joint_pos_cmd[v]
            command["joint_vel"][k] = joint_vel_cmd[v]
            command["joint_trq"][k] = joint_trq_cmd[v]

        return command

    def update_system(
        self,
        base_com_pos,
        base_com_quat,
        base_com_lin_vel,
        base_com_ang_vel,
        base_joint_pos,
        base_joint_quat,
        base_joint_lin_vel,
        base_joint_ang_vel,
        joint_pos,
        joint_vel,
        b_cent=True,
    ):
        # assert len(joint_pos.keys()) == self._n_a

        self._q = np.zeros(self._n_q)
        self._q_dot = np.zeros(self._n_q_dot)
        self._joint_positions = np.zeros(self._n_a)
        self._joint_velocities = np.zeros(self._n_a)
        if not self._b_fixed_base:
            # Floating Based Robot
            self._q[0:3] = np.copy(base_joint_pos)
            self._q[3:7] = np.copy(base_joint_quat)

            rot_w_basejoint = geom.quat_to_rot(base_joint_quat)
            twist_basejoint_in_world = np.zeros(6)
            twist_basejoint_in_world[0:3] = base_joint_ang_vel
            twist_basejoint_in_world[3:6] = base_joint_lin_vel
            augrot_joint_world = np.zeros((6, 6))
            augrot_joint_world[0:3, 0:3] = rot_w_basejoint.transpose()
            augrot_joint_world[3:6, 3:6] = rot_w_basejoint.transpose()
            twist_basejoint_in_joint = np.dot(
                augrot_joint_world, twist_basejoint_in_world
            )
            self._q_dot[0:3] = twist_basejoint_in_joint[3:6]
            self._q_dot[3:6] = twist_basejoint_in_joint[0:3]
        else:
            # Fixed Based Robot
            pass

        self._q[self.get_q_idx(list(joint_pos.keys()))] = np.copy(
            list(joint_pos.values())
        )
        self._q_dot[self.get_q_dot_idx(list(joint_vel.keys()))] = np.copy(
            list(joint_vel.values())
        )
        self._joint_positions[self.get_joint_idx(list(joint_pos.keys()))] = np.copy(
            list(joint_pos.values())
        )
        self._joint_velocities[self.get_joint_idx(list(joint_vel.keys()))] = np.copy(
            list(joint_vel.values())
        )

        pin.forwardKinematics(self._model, self._data, self._q, self._q_dot)

        if b_cent:
            self._update_centroidal_quantities()

    def _update_centroidal_quantities(self):
        pin.ccrba(self._model, self._data, self._q, self._q_dot)

        self._hg = np.zeros_like(self._data.hg)
        self._hg[0:3] = np.copy(self._data.hg.angular)
        self._hg[3:6] = np.copy(self._data.hg.linear)

        self._Ag = np.zeros_like(self._data.Ag)
        self._Ag[0:3] = np.copy(self._data.Ag[3:6, :])
        self._Ag[3:6] = np.copy(self._data.Ag[0:3, :])

        self._Ig = np.zeros_like(self._data.Ig)
        self._Ig[0:3, 0:3] = np.copy(self._data.Ig)[3:6, 3:6]
        self._Ig[3:6, 3:6] = np.copy(self._data.Ig)[0:3, 0:3]

    def get_q(self):
        return np.copy(self._q)

    def get_q_dot(self):
        return np.copy(self._q_dot)

    def get_mass_matrix(self):
        return np.copy(pin.crba(self._model, self._data, self._q))

    def get_gravity(self):
        return np.copy(pin.computeGeneralizedGravity(self._model, self._data, self._q))

    def get_coriolis(self):
        return np.copy(
            pin.nonLinearEffects(self._model, self._data, self._q, self._q_dot)
            - self.get_gravity()
        )

    def get_com_pos(self):
        pin.centerOfMass(self._model, self._data, self._q, self._q_dot)
        return np.copy(self._data.com[0])

    def get_com_lin_vel(self):
        pin.centerOfMass(self._model, self._data, self._q, self._q_dot)
        return np.copy(self._data.vcom[0])

    def get_com_lin_jacobian(self):
        return np.copy(pin.jacobianCenterOfMass(self._model, self._data, self._q))

    def get_com_lin_jacobian_dot(self):
        return np.copy(
            (
                pin.computeCentroidalMapTimeVariation(
                    self._model, self._data, self._q, self._q_dot
                )[0:3, :]
            )
            / self._total_mass
        )

    def get_link_iso(self, link_id):
        ret = np.eye(4)
        frame_id = self._model.getFrameId(link_id)
        trans = pin.updateFramePlacement(self._model, self._data, frame_id)
        ret[0:3, 0:3] = trans.rotation
        ret[0:3, 3] = trans.translation
        return np.copy(ret)

    def get_link_vel(self, link_id):
        ret = np.zeros(6)
        frame_id = self._model.getFrameId(link_id)

        spatial_vel = pin.getFrameVelocity(
            self._model, self._data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        ret[0:3] = spatial_vel.angular
        ret[3:6] = spatial_vel.linear

        return np.copy(ret)

    def get_link_jacobian(self, link_id):
        frame_id = self._model.getFrameId(link_id)
        pin.computeJointJacobians(self._model, self._data, self._q)
        jac = pin.getFrameJacobian(
            self._model, self._data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # Pinocchio has linear on top of angular
        ret = np.zeros_like(jac)
        ret[0:3] = jac[3:6]
        ret[3:6] = jac[0:3]

        return np.copy(ret)

    def get_link_jacobian_dot_times_qdot(self, link_id):
        frame_id = self._model.getFrameId(link_id)

        pin.forwardKinematics(
            self._model, self._data, self._q, self._q_dot, 0 * self._q_dot
        )
        jdot_qdot = pin.getFrameClassicalAcceleration(
            self._model, self._data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        ret = np.zeros_like(jdot_qdot)
        ret[0:3] = jdot_qdot.angular
        ret[3:6] = jdot_qdot.linear

        return np.copy(ret)

    def get_Ag(self):
        return np.copy(self._Ag)

    def get_Ig(self):
        return np.copy(self._Ig)

    def get_hg(self):
        return np.copy(self._hg)
