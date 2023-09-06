import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
from collections import OrderedDict
import dartpy as dart
import numpy as np

from .base import RobotSystem
from util import geom as geom


class DartRobotSystem(RobotSystem):
    """
    Dart considers floating base with 6 positions and 6 velocities with the
    order of rotx, roty, rotz, x, y, z. Therefore, n_q = n_v
    Note that the joint named with 'rootJoint' has 6 dof to represent the
    floating base.
    """
    def __init__(self, urdf_file, b_fixed_base, b_print_info=False):
        super(DartRobotSystem, self).__init__(urdf_file, None, b_fixed_base,
                                              b_print_info)

    def _config_robot(self, urdf_file, package_dir):
        self._skel = dart.utils.DartLoader().parseSkeleton(urdf_file)

        for i in range(self._skel.getNumJoints()):
            j = self._skel.getJoint(i)
            if j.getName() == 'rootJoint':
                self._n_floating = j.getNumDofs()
                assert (not self._b_fixed_base)
                assert self._n_floating == 6
            elif j.getType() != "WeldJoint":
                self._joint_id[j.getName()] = j
            else:
                pass

        for i in range(self._skel.getNumBodyNodes()):
            bn = self._skel.getBodyNode(i)
            self._link_id[bn.getName()] = bn

        self._n_q = self._n_q_dot = self._skel.getNumDofs()
        self._n_a = self._n_q_dot - self._n_floating
        self._total_mass = self._skel.getMass()
        self._joint_pos_limit = np.stack(
            [
                self._skel.getPositionLowerLimits(),
                self._skel.getPositionUpperLimits()
            ],
            axis=1)[self._n_floating:self._n_floating + self._n_a, :]
        self._joint_vel_limit = np.stack(
            [
                self._skel.getVelocityLowerLimits(),
                self._skel.getVelocityUpperLimits()
            ],
            axis=1)[self._n_floating:self._n_floating + self._n_a, :]
        self._joint_trq_limit = np.stack(
            [
                self._skel.getForceLowerLimits(),
                self._skel.getForceUpperLimits()
            ],
            axis=1)[self._n_floating:self._n_floating + self._n_a, :]

    def get_q_idx(self, joint_id):
        if type(joint_id) is list:
            return [
                self._joint_id[id].getIndexInSkeleton(0) for id in joint_id
            ]
        else:
            return self._joint_id[joint].getIndexInSkeleton(0)

    def get_q_dot_idx(self, joint_id):
        if type(joint_id) is list:
            return [
                self._joint_id[id].getIndexInSkeleton(0) for id in joint_id
            ]
        else:
            return self._joint_id[joint].getIndexInSkeleton(0)

    def get_joint_idx(self, joint_id):
        if type(joint_id) is list:
            return [self.get_joint_idx(j_id) for j_id in joint_id]
        else:
            return self._joint_id[joint_id].getIndexInSkeleton(
                0) - self._n_floating

    def create_cmd_ordered_dict(self, joint_pos_cmd, joint_vel_cmd,
                                joint_trq_cmd):

        command = OrderedDict()
        command["joint_pos"] = OrderedDict()
        command["joint_vel"] = OrderedDict()
        command["joint_trq"] = OrderedDict()

        for k, v in self._joint_id.items():
            joint_idx = self._joint_id[k].getIndexInSkeleton(
                0) - self._n_floating
            command["joint_pos"][k] = joint_pos_cmd[joint_idx]
            command["joint_vel"][k] = joint_vel_cmd[joint_idx]
            command["joint_trq"][k] = joint_trq_cmd[joint_idx]

        return command

    def update_system(self,
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
                      b_cent=False):

        assert len(joint_pos.keys()) == self._n_a

        if not self._b_fixed_base:
            # Floating Base Robot
            joint_iso = dart.math.Isometry3()
            joint_iso.set_rotation(geom.quat_to_rot(base_joint_quat))
            joint_iso.set_translation(base_joint_pos)

            joint_vel_in_world = np.zeros(6)
            joint_vel_in_world[0:3] = base_joint_ang_vel
            joint_vel_in_world[3:6] = base_joint_lin_vel
            self._skel.getRootJoint().setSpatialMotion(
                joint_iso, dart.dynamics.Frame.World(),
                np.reshape(joint_vel_in_world, (6, 1)),
                dart.dynamics.Frame.World(), dart.dynamics.Frame.World(),
                np.zeros((6, 1)), dart.dynamics.Frame.World(),
                dart.dynamics.Frame.World())
        else:
            # Fixed Base Robot
            pass

        self._joint_positions = np.zeros(self._n_a)
        self._joint_velocities = np.zeros(self._n_a)
        for (p_k, p_v), (v_k, v_v) in zip(joint_pos.items(),
                                          joint_vel.items()):
            # Assume the joints have 1 dof
            self._joint_id[p_k].setPosition(0, p_v)
            self._joint_id[v_k].setVelocity(0, v_v)

            self._joint_positions[self.get_joint_idx(p_k)] = p_v
            self._joint_velocities[self.get_joint_idx(v_k)] = v_v

        self._skel.computeForwardKinematics()

        if b_cent:
            self._update_centroidal_quantities()

    def _update_centroidal_quantities(self):
        self._Ig = np.zeros((6, 6))
        self._Ag = np.zeros((6, self._n_q_dot))
        pCoM_g = self.get_com_pos()

        for name, bn in self._link_id.items():
            jac = self._skel.getJacobian(bn)
            p_gl = bn.getWorldTransform().translation()
            R_gl = bn.getWorldTransform().rotation()
            I = bn.getInertia().getSpatialTensor()
            T_lc = np.eye(4)
            T_lc[0:3, 0:3] = R_gl.transpose()
            T_lc[0:3, 3] = np.dot(R_gl.transpose(), (pCoM_g - p_gl))
            AdT_lc = geom.adjoint(T_lc)
            self._Ig += np.dot(np.dot(AdT_lc.transpose(), I), AdT_lc)
            self._Ag += np.dot(np.dot(AdT_lc.transpose(), I), jac)

        self._hg = np.dot(self._Ag, self.get_q_dot())

    def get_q(self):
        return np.copy(self._skel.getPositions())

    def get_q_dot(self):
        return np.copy(self._skel.getVelocities())

    def get_mass_matrix(self):
        return np.copy(self._skel.getMassMatrix())

    def get_gravity(self):
        return np.copy(self._skel.getGravityForces())

    def get_coriolis(self):
        return np.copy(self._skel.getCoriolisForces())

    def get_com_pos(self):
        return np.copy(self._skel.getCOM(dart.dynamics.Frame.World()))

    def get_com_lin_vel(self):
        return np.copy(
            self._skel.getCOMLinearVelocity(dart.dynamics.Frame.World(),
                                            dart.dynamics.Frame.World()))

    def get_com_lin_jacobian(self):
        return np.copy(
            self._skel.getCOMLinearJacobian(dart.dynamics.Frame.World()))

    def get_com_lin_jacobian_dot(self):
        return np.copy(
            self._skel.getCOMLinearJacobianDeriv(dart.dynamics.Frame.World()))

    def get_link_iso(self, link_id):
        link_iso = self._link_id[link_id].getTransform(
            dart.dynamics.Frame.World(), dart.dynamics.Frame.World())
        ret = np.eye(4)
        ret[0:3, 0:3] = link_iso.rotation()
        ret[0:3,
            3] = self._link_id[link_id].getCOM(dart.dynamics.Frame.World())
        return np.copy(ret)

    def get_link_vel(self, link_id):
        return np.copy(self._link_id[link_id].getCOMSpatialVelocity(
            dart.dynamics.Frame.World(), dart.dynamics.Frame.World()))

    def get_link_jacobian(self, link_id):
        return np.copy(
            self._skel.getJacobian(self._link_id[link_id],
                                   self._link_id[link_id].getLocalCOM(),
                                   dart.dynamics.Frame.World()))

    def get_link_jacobian_dot_times_qdot(self, link_id):

        jac_dot = self._skel.getJacobianClassicDeriv(
            self._link_id[link_id], self._link_id[link_id].getLocalCOM(),
            dart.dynamics.Frame.World())

        return np.dot(jac_dot, self.get_q_dot())

    def get_Ag(self):
        return np.copy(self._Ag)

    def get_Ig(self):
        return np.copy(self._Ig)
    
    def get_hg(self):
        return np.copy(self._hg)