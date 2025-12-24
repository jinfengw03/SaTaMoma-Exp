import os
import numpy as np
from tracikpy import TracIKSolver
from tiago_client.utils.transformations import quat_to_rmat

class TiagoIK:
    def __init__(self, side='right'):
        self.side = side
        self.urdf_path = os.path.join('/home/jiachenli/SaTaMoma-Exp/tiago_client/tiago_client', f'urdf/tiago_{self.side}_arm.urdf')
        
        # Initialize IK Solver
        self.ik_solver = TracIKSolver(
            urdf_file=self.urdf_path,
            base_link="torso_lift_link",
            tip_link=f"arm_{self.side}_7_link",
            timeout=0.025,
            epsilon=5e-4,
            solve_type="Distance"
        )

    def find_ik(self, target_pos, target_quat, joint_init):
        """
        Calculates joint angles for a target Cartesian pose.
        :param target_pos: [x, y, z]
        :param target_quat: [x, y, z, w]
        :param joint_init: Current joint angles for seeding the solver
        :return: joint_goal (7 DOF) or None
        """
        ee_pose = np.eye(4)
        ee_pose[:3, :3] = quat_to_rmat(target_quat)
        ee_pose[:3, 3] = np.array(target_pos)
        
        joint_goal = self.ik_solver.ik(ee_pose, qinit=joint_init)
        return joint_goal
