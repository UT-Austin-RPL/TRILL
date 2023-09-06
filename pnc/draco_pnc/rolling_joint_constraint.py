import numpy as np

from pnc.wbc.internal_constraint import InternalConstraint


class DracoManipulationRollingJointConstraint(InternalConstraint):
    def __init__(self, robot):
        super(DracoManipulationRollingJointConstraint, self).__init__(robot, 2)
        l_jp_idx, l_jd_idx, r_jp_idx, r_jd_idx = self._robot.get_q_dot_idx(
            ['l_knee_fe_jp', 'l_knee_fe_jd', 'r_knee_fe_jp', 'r_knee_fe_jd'])

        self._jacobian[0, l_jp_idx] = 1.
        self._jacobian[0, l_jd_idx] = -1.
        self._jacobian[1, r_jp_idx] = 1.
        self._jacobian[1, r_jd_idx] = -1.
        self._jacobian_dot_q_dot = np.zeros(2)

    def _update_jacobian(self):
        # Constant jacobian
        pass
