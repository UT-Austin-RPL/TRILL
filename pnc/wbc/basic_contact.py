import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np

from pnc.wbc.contact import Contact
from pnc.data_saver import DataSaver


class PointContact(Contact):
    def __init__(self, robot, link_id, mu, data_save=False):
        super(PointContact, self).__init__(robot, 3)

        self._link_id = link_id
        self._mu = mu
        self._b_data_save = data_save

        if self._b_data_save:
            self._data_saver = DataSaver()

    def _update_jacobian(self):
        self._jacobian = self._robot.get_link_jacobian(self._link_id)[
            self._dim_contact :, :
        ]
        self._jacobian_dot_q_dot = self._robot.get_link_jacobian_dot_times_qdot(
            self._link_id
        )[self._dim_contact :]

    def _update_cone_constraint(self):
        rot = self._robot.get_link_iso(self._link_id)[0:3, 0:3].transpose()
        self._cone_constraint_mat = np.zeros((6, self._dim_contact))
        self._cone_constraint_mat[0, 2] = 1.0

        self._cone_constraint_mat[1, 0] = 1.0
        self._cone_constraint_mat[1, 2] = self._mu
        self._cone_constraint_mat[2, 0] = -1.0
        self._cone_constraint_mat[2, 2] = self._mu

        self._cone_constraint_mat[3, 1] = 1.0
        self._cone_constraint_mat[3, 2] = self._mu
        self._cone_constraint_mat[4, 1] = -1.0
        self._cone_constraint_mat[4, 2] = self._mu

        self._cone_constraint_mat[5, 2] = -1.0

        self._cone_constraint_vec = np.zeros(6)
        self._cone_constraint_vec[5] = -self._rf_z_max

        if self._b_data_save:
            self._data_saver.add("rf_z_max_" + self._link_id, self._rf_z_max)


class SurfaceContact(Contact):
    def __init__(self, robot, link_id, x, y, mu, data_save=False):
        super(SurfaceContact, self).__init__(robot, 6)

        self._link_id = link_id
        self._x = x
        self._y = y
        self._mu = mu
        self._b_data_save = data_save

        if self._b_data_save:
            self._data_saver = DataSaver()

    def _update_jacobian(self):
        self._jacobian = self._robot.get_link_jacobian(self._link_id)
        self._jacobian_dot_q_dot = self._robot.get_link_jacobian_dot_times_qdot(
            self._link_id
        )

    def _update_cone_constraint(self):
        self._cone_constraint_mat = np.zeros((16 + 2, self._dim_contact))

        u = self._get_u(self._x, self._y, self._mu)
        rot = self._robot.get_link_iso(self._link_id)[0:3, 0:3]
        rot_foot = np.zeros((6, 6))
        rot_foot[0:3, 0:3] = rot.transpose()
        rot_foot[3:6, 3:6] = rot.transpose()

        self._cone_constraint_mat = np.dot(u, rot_foot)

        self._cone_constraint_vec = np.zeros(16 + 2)
        self._cone_constraint_vec[17] = -self._rf_z_max

        if self._b_data_save:
            self._data_saver.add("rf_z_max_" + self._link_id, self._rf_z_max)

    def _get_u(self, x, y, mu):
        u = np.zeros((16 + 2, 6))

        u[0, 5] = 1.0

        u[1, 3] = 1.0
        u[1, 5] = mu
        u[2, 3] = -1.0
        u[2, 5] = mu

        u[3, 4] = 1.0
        u[3, 5] = mu
        u[4, 4] = -1.0
        u[4, 5] = mu

        u[5, 0] = 1.0
        u[5, 5] = y
        u[6, 0] = -1.0
        u[6, 5] = y

        u[7, 1] = 1.0
        u[7, 5] = x
        u[8, 1] = -1.0
        u[8, 5] = x

        ##tau
        u[9, 0] = -mu
        u[9, 1] = -mu
        u[9, 2] = 1.0
        u[9, 3] = y
        u[9, 4] = x
        u[9, 5] = (x + y) * mu

        u[10, 0] = -mu
        u[10, 1] = mu
        u[10, 2] = 1.0
        u[10, 3] = y
        u[10, 4] = -x
        u[10, 5] = (x + y) * mu

        u[11, 0] = mu
        u[11, 1] = -mu
        u[11, 2] = 1.0
        u[11, 3] = -y
        u[11, 4] = x
        u[11, 5] = (x + y) * mu

        u[12, 0] = mu
        u[12, 1] = mu
        u[12, 2] = 1.0
        u[12, 3] = -y
        u[12, 4] = -x
        u[12, 5] = (x + y) * mu

        u[13, 0] = -mu
        u[13, 1] = -mu
        u[13, 2] = -1.0
        u[13, 3] = -y
        u[13, 4] = -x
        u[13, 5] = (x + y) * mu

        u[14, 0] = -mu
        u[14, 1] = mu
        u[14, 2] = -1.0
        u[14, 3] = -y
        u[14, 4] = x
        u[14, 5] = (x + y) * mu

        u[15, 0] = mu
        u[15, 1] = -mu
        u[15, 2] = -1.0
        u[15, 3] = y
        u[15, 4] = -x
        u[15, 5] = (x + y) * mu

        u[16, 0] = mu
        u[16, 1] = mu
        u[16, 2] = -1.0
        u[16, 3] = y
        u[16, 4] = x
        u[16, 5] = (x + y) * mu

        u[17, 5] = -1.0

        return u
