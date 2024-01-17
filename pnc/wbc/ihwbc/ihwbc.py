import sys

import numpy as np
from qpsolvers import solve_qp
from scipy.linalg import block_diag

from pnc.data_saver import DataSaver
from util.linalg import weighted_pinv

np.set_printoptions(precision=2, threshold=sys.maxsize)


class IHWBC(object):
    """
    Implicit Hierarchy Whole Body Control
    ------------------
    Usage:
        update_setting --> solve
    """

    def __init__(self, sf, sa, sv, data_save=False):
        self._n_q_dot = sa.shape[1]
        self._n_active = sa.shape[0]
        self._n_passive = sv.shape[0]

        self._sf = sf
        self._snf = np.concatenate(
            (
                np.zeros((self._n_active + self._n_passive, 6)),
                np.eye(self._n_active + self._n_passive),
            ),
            axis=1,
        )
        self._sa = sa
        self._sv = sv

        self._trq_limit = None
        self._lambda_q_ddot = 0.0
        self._lambda_rf = 0.0
        self._w_rf = 0.0
        self._w_hierarchy = 0.0

        self._b_data_save = data_save
        if self._b_data_save:
            self._data_saver = DataSaver()

    @property
    def trq_limit(self):
        return self._trq_limit

    @property
    def lambda_q_ddot(self):
        return self._lambda_q_ddot

    @property
    def lambda_rf(self):
        return self._lambda_rf

    @property
    def w_hierarchy(self):
        return self._w_hierarchy

    @property
    def w_rf(self):
        return self._w_rf

    @trq_limit.setter
    def trq_limit(self, val):
        assert val.shape[0] == self._n_active
        self._trq_limit = np.copy(val)

    @lambda_q_ddot.setter
    def lambda_q_ddot(self, val):
        self._lambda_q_ddot = val

    @lambda_rf.setter
    def lambda_rf(self, val):
        self._lambda_rf = val

    @w_hierarchy.setter
    def w_hierarchy(self, val):
        self._w_hierarchy = val

    @w_hierarchy.setter
    def w_rf(self, val):
        self._w_rf = val

    def update_setting(self, mass_matrix, mass_matrix_inv, coriolis, gravity):
        self._mass_matrix = np.copy(mass_matrix)
        self._mass_matrix_inv = np.copy(mass_matrix_inv)
        self._coriolis = np.copy(coriolis)
        self._gravity = np.copy(gravity)

    def solve(
        self,
        task_list,
        contact_list,
        internal_constraint_list,
        rf_des=None,
        verbose=False,
    ):
        """
        Parameters
        ----------
        task_list (list of Task):
            Task list
        contact_list (list of Contact):
            Contact list
        internal_constraint_list (list of InternalConstraint):
            Internal constraint list
        rf_des (np.ndarray):
            Reaction force desired
        verbose (bool):
            Printing option

        Returns
        -------
        joint_trq_cmd (np.array):
            Joint trq cmd
        joint_acc_cmd (np.array):
            Joint acc cmd
        sol_rf (np.array):
            Reaction force
        """

        # ======================================================================
        # Internal Constraint
        #   Set ni, jit_lmd_jidot_qdot, sa_ni_trc_bar_tr, and b_internal_constraint
        # ======================================================================
        if len(internal_constraint_list) > 0:
            ji = np.concatenate(
                [ic.jacobian for ic in internal_constraint_list], axis=0
            )
            jidot_qdot = np.concatenate(
                [ic.jacobian_dot_q_dot for ic in internal_constraint_list], axis=0
            )
            lmd = np.linalg.pinv(
                np.dot(np.dot(ji, self._mass_matrix_inv), ji.transpose())
            )
            ji_bar = np.dot(np.dot(self._mass_matrix_inv, ji.transpose()), lmd)
            ni = np.eye(self._n_q_dot) - np.dot(ji_bar, ji)
            jit_lmd_jidot_qdot = np.squeeze(
                np.dot(np.dot(ji.transpose(), lmd), jidot_qdot)
            )
            sa_ni_trc = np.dot(self._sa, ni)[:, 6:]
            sa_ni_trc_bar = weighted_pinv(sa_ni_trc, self._mass_matrix_inv[6:, 6:])
            sa_ni_trc_bar_tr = sa_ni_trc_bar.transpose()
            b_internal_constraint = True
        else:
            ni = np.eye(self._n_q_dot)
            jit_lmd_jidot_qdot = np.zeros(self._n_q_dot)
            sa_ni_trc_bar = np.eye(self._n_active)
            sa_ni_trc_bar_tr = sa_ni_trc_bar.transpose()
            b_internal_constraint = False

        # print("ni")
        # print(ni)
        # print("jit_lmd_jidot_qdot")
        # print(jit_lmd_jidot_qdot)
        # print("sa_ni_trc_bar_tr")
        # print(sa_ni_trc_bar_tr)
        # exit()

        # ======================================================================
        # Cost
        # ======================================================================
        cost_t_mat = np.zeros((self._n_q_dot, self._n_q_dot))
        cost_t_vec = np.zeros(self._n_q_dot)

        for i, task in enumerate(task_list):
            j = task.jacobian
            j_dot_q_dot = task.jacobian_dot_q_dot
            x_ddot = task.op_cmd
            if verbose:
                print("====================")
                print(task.target_id, " task")
                task.debug()

            cost_t_mat += self._w_hierarchy[i] * np.dot(j.transpose(), j)
            cost_t_vec += self._w_hierarchy[i] * np.dot(
                (j_dot_q_dot - x_ddot).transpose(), j
            )
        # cost_t_mat += self._lambda_q_ddot * np.eye(self._n_q_dot)
        cost_t_mat += self._lambda_q_ddot * self._mass_matrix

        if contact_list is not None:
            uf_mat = np.array(
                block_diag(*[contact.cone_constraint_mat for contact in contact_list])
            )
            uf_vec = np.concatenate(
                [contact.cone_constraint_vec for contact in contact_list]
            )
            contact_jacobian = np.concatenate(
                [contact.jacobian for contact in contact_list], axis=0
            )

            assert uf_mat.shape[0] == uf_vec.shape[0]
            assert uf_mat.shape[1] == contact_jacobian.shape[0]
            dim_cone_constraint, dim_contacts = uf_mat.shape

            cost_rf_mat = (self._lambda_rf + self._w_rf) * np.eye(dim_contacts)
            if rf_des is None:
                rf_des = np.zeros(dim_contacts)
            cost_rf_vec = -self._w_rf * np.copy(rf_des)

            cost_mat = np.array(
                block_diag(cost_t_mat, cost_rf_mat)
            )  # (nqdot+nc, nqdot+nc)
            cost_vec = np.concatenate([cost_t_vec, cost_rf_vec])  # (nqdot+nc,)

        else:
            dim_contacts = dim_cone_constraint = 0
            cost_mat = np.copy(cost_t_mat)
            cost_vec = np.copy(cost_t_vec)

        # if verbose:
        # print("==================================")
        # np.set_printoptions(precision=4)
        # print("cost_t_mat")
        # print(cost_t_mat)
        # print("cost_t_vec")
        # print(cost_t_vec)
        # print("cost_rf_mat")
        # print(cost_rf_mat)
        # print("cost_rf_vec")
        # print(cost_rf_vec)
        # print("cost_mat")
        # print(cost_mat)
        # print("cost_vec")
        # print(cost_vec)

        # ======================================================================
        # Equality Constraint
        # ======================================================================

        if contact_list is not None:
            eq_floating_mat = np.concatenate(
                (
                    np.dot(self._sf, self._mass_matrix),
                    -np.dot(self._sf, np.dot(contact_jacobian, ni).transpose()),
                ),
                axis=1,
            )  # (6, nqdot+nc)
            if b_internal_constraint:
                eq_int_mat = np.concatenate(
                    (ji, np.zeros((ji.shape[0], dim_contacts))), axis=1
                )  # (2, nqdot+nc)
                eq_int_vec = np.zeros(ji.shape[0])
        else:
            eq_floating_mat = np.dot(self._sf, self._mass_matrix)
            if b_internal_constraint:
                eq_int_mat = np.copy(ji)
                eq_int_vec = np.zeros(ji.shape[0])
        eq_floating_vec = -np.dot(
            self._sf, np.dot(ni.transpose(), (self._coriolis + self._gravity))
        )

        if b_internal_constraint:
            eq_mat = np.concatenate((eq_floating_mat, eq_int_mat), axis=0)
            eq_vec = np.concatenate((eq_floating_vec, eq_int_vec), axis=0)
        else:
            eq_mat = np.copy(eq_floating_mat)
            eq_vec = np.copy(eq_floating_vec)

        # ======================================================================
        # Inequality Constraint
        # ======================================================================

        if self._trq_limit is None:
            if contact_list is not None:
                ineq_mat = np.concatenate(
                    (np.zeros((dim_cone_constraint, self._n_q_dot)), -uf_mat), axis=1
                )
                ineq_vec = -uf_vec
            else:
                ineq_mat = None
                ineq_vec = None

        else:
            if contact_list is not None:
                ineq_mat = np.concatenate(
                    (
                        np.concatenate(
                            (
                                np.zeros((dim_cone_constraint, self._n_q_dot)),
                                -np.dot(
                                    sa_ni_trc_bar_tr,
                                    np.dot(self._snf, self._mass_matrix),
                                ),
                                np.dot(
                                    sa_ni_trc_bar_tr,
                                    np.dot(self._snf, self._mass_matrix),
                                ),
                            ),
                            axis=0,
                        ),
                        np.concatenate(
                            (
                                -uf_mat,
                                np.dot(
                                    np.dot(sa_ni_trc_bar_tr, self._snf),
                                    np.dot(contact_jacobian, ni).transpose(),
                                ),
                                -np.dot(
                                    np.dot(sa_ni_trc_bar_tr, self._snf),
                                    np.dot(contact_jacobian, ni).transpose(),
                                ),
                            ),
                            axis=0,
                        ),
                    ),
                    axis=1,
                )
                ineq_vec = np.concatenate(
                    (
                        -uf_vec,
                        np.dot(
                            np.dot(sa_ni_trc_bar_tr, self._snf),
                            np.dot(ni.transpose(), (self._coriolis + self._gravity)),
                        )
                        + np.dot(
                            np.dot(sa_ni_trc_bar_tr, self._snf), jit_lmd_jidot_qdot
                        )
                        - self._trq_limit[:, 0],
                        -np.dot(
                            np.dot(sa_ni_trc_bar_tr, self._snf),
                            np.dot(ni.transpose(), (self._coriolis + self._gravity)),
                        )
                        - np.dot(
                            np.dot(sa_ni_trc_bar_tr, self._snf), jit_lmd_jidot_qdot
                        )
                        + self._trq_limit[:, 1],
                    )
                )

            else:
                ineq_mat = np.concatenate(
                    (
                        -np.dot(np.dot(sa_ni_trc_bar_tr, self._snf), self._mass_matrix),
                        np.dot(np.dot(sa_ni_trc_bar_tr, self._snf), self._mass_matrix),
                    ),
                    axis=0,
                )
                ineq_vec = np.concatenate(
                    (
                        np.dot(
                            np.dot(sa_ni_trc_bar_tr, self._snf),
                            np.dot(ni.transpose(), (self._coriolis + self._gravity)),
                        )
                        + np.dot(
                            np.dot(sa_ni_trc_bar_tr, self._snf), jit_lmd_jidot_qdot
                        )
                        - self._trq_limit[:, 0],
                        -np.dot(
                            np.dot(sa_ni_trc_bar_tr, self._snf),
                            np.dot(ni.transpose(), (self._coriolis + self._gravity)),
                        )
                        - np.dot(
                            np.dot(sa_ni_trc_bar_tr, self._snf), jit_lmd_jidot_qdot
                        )
                        + self._trq_limit[:, 1],
                    )
                )

        # if verbose:
        # print("eq_mat")
        # print(eq_mat)
        # print("eq_vec")
        # print(eq_vec)

        # print("ineq_mat")
        # print(ineq_mat)
        # print("ineq_vec")
        # print(ineq_vec)

        sol = solve_qp(
            cost_mat,
            cost_vec,
            ineq_mat,
            ineq_vec,
            eq_mat,
            eq_vec,
            solver="quadprog",
            verbose=True,
        )

        if contact_list is not None:
            sol_q_ddot, sol_rf = sol[: self._n_q_dot], sol[self._n_q_dot :]
        else:
            sol_q_ddot, sol_rf = sol, None

        if contact_list is not None:
            joint_trq_cmd = np.dot(
                np.dot(sa_ni_trc_bar_tr, self._snf),
                np.dot(self._mass_matrix, sol_q_ddot)
                + np.dot(ni.transpose(), (self._coriolis + self._gravity))
                - np.dot(np.dot(contact_jacobian, ni).transpose(), sol_rf),
            )
        else:
            joint_trq_cmd = np.dot(
                np.dot(sa_ni_trc_bar_tr, self._snf),
                np.dot(self._mass_matrix, sol_q_ddot)
                + np.dot(ni, (self._coriolis + self._gravity)),
            )

        joint_acc_cmd = np.dot(self._sa, sol_q_ddot)

        if verbose:
            # if True:
            print("joint_trq_cmd: ", joint_trq_cmd)
            print("sol_q_ddot: ", sol_q_ddot)
            print("sol_rf: ", sol_rf)

            # for i, task in enumerate(task_list):
            for task in [task_list[3], task_list[4]]:
                j = task.jacobian
                j_dot_q_dot = task.jacobian_dot_q_dot
                x_ddot = task.op_cmd
                print(task.target_id, " task")
                print("des x ddot: ", x_ddot)
                print("j*qddot_sol + Jdot*qdot: ", np.dot(j, sol_q_ddot) + j_dot_q_dot)

        if self._b_data_save:
            self._data_saver.add("joint_trq_cmd", joint_trq_cmd)
            self._data_saver.add("joint_acc_cmd", joint_acc_cmd)
            self._data_saver.add("rf_cmd", sol_rf)

        return joint_trq_cmd, joint_acc_cmd, sol_rf
