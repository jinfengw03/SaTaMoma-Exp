import os
import rospy
import numpy as np
from std_msgs.msg import Header
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tiago_server.tiago_sys.utils.ros_utils import Publisher, Listener, TFTransformListener
from tiago_server.tiago_sys.utils.transformations import euler_to_quat, quat_to_euler, quat_diff, add_angles, add_quats, quat_to_rmat
from tracikpy import TracIKSolver # repo for Inverse Kinematics computation  


class TiagoArms:
    
    def __init__(self, arm_enabled,  side='right') -> None:
        self.arm_enabled = arm_enabled
        self.side = side
        self.urdf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../urdf/tiago.urdf')
        self.setup_listeners()
        self.setup_actors()
        # self._internal_state_pose = np.zeros(3) # x|y|z <-> cartesian space
        # self._internal_state_quat = np.zeros(4) # x|y|z|w <-> cartesian space

    def setup_listeners(self):
        def joint_process_func(data):
            return np.array(data.actual.positions)
        self.arm_reader = TFTransformListener('/base_footprint') # default 'base_link' is '/base_footprint'
        self.joint_reader = Listener(f'/arm_{self.side}_controller/state', JointTrajectoryControllerState, post_process_func=joint_process_func)
    
    @property
    def arm_pose(self):
        pos, quat = self.arm_reader.get_transform(target_link=f'/arm_{self.side}_tool_link')
        if pos is None:
            return None
        return np.concatenate((pos, quat))
    
    def setup_actors(self):
        self.arm_writer = None
        if self.arm_enabled:
            self.ik_solver = TracIKSolver(
                urdf_file=self.urdf_path,
                base_link="torso_lift_link",
                tip_link=f"arm_{self.side}_tool_link",
                timeout=0.025,
                epsilon=5e-4,
                solve_type="Distance"
                ) # Inverse Kinematics Solver
            self.arm_writer = Publisher(f'/arm_{self.side}_controller/safe_command', JointTrajectory)
    
    def process_action(self, action):
        # convert deltas to absolute positions
        pos_delta, euler_delta = action[:3], action[3:6]
        cur_pos, cur_quat = self.arm_reader.get_transform(target_link=f'/arm_{self.side}_tool_link', base_link='/torso_lift_link')
        cur_euler = quat_to_euler(cur_quat)
        target_pos = cur_pos + pos_delta
        target_euler = add_angles(euler_delta, cur_euler)
        target_quat = euler_to_quat(target_euler)
        return target_pos, target_quat
    
    def create_joint_command(self, joint_goal, duration_scale):
        message = JointTrajectory()
        message.header = Header()
        joint_names = []
        positions = list(self.joint_reader.get_most_recent_msg())
        for i in range(1, 8):
            joint_names.append(f'arm_{self.side}_{i}_joint')
            positions[i-1] = joint_goal[i-1]  
        message.joint_names = joint_names
        # duration = 1.3 
        duration = 0.7 + duration_scale
        point = JointTrajectoryPoint(positions=positions, time_from_start = rospy.Duration(duration))
        message.points.append(point)
        return message 
    
    def write(self, joint_goal, duration_scale):
        pose_command = self.create_joint_command(joint_goal, duration_scale)
        if self.arm_writer is not None:
            self.arm_writer.write(pose_command)
            pass
    
    def find_ik(self, target_pos, target_quat):
        ee_pose = np.eye(4)
        ee_pose[:3, :3] = quat_to_rmat(target_quat)
        ee_pose[:3, 3] = np.array(target_pos)
        joint_init = self.joint_reader.get_most_recent_msg()
        joint_goal = self.ik_solver.ik(ee_pose, qinit=joint_init)
        duration_scale = 0
        if joint_goal is not None:
            duration_scale = np.linalg.norm(joint_goal-joint_init)
        return joint_goal, duration_scale
    
    def step(self, action):
        if self.arm_enabled:
            assert len(action) == 6 # x|y|z + rx|ry|rz
            target_pos, target_quat = self.process_action(action) # x|y|z, x|y|z|w
            joint_goal, duration_scale = self.find_ik(target_pos, target_quat)
            if joint_goal is not None:
                self.write(joint_goal, duration_scale) # [absolute control]

    def step_joints(self, joint_goal):
        if self.arm_enabled:
            assert len(joint_goal) == 7
            # Calculate duration_scale based on distance from current joints
            cur_joints = self.joint_reader.get_most_recent_msg()
            duration_scale = np.linalg.norm(joint_goal - cur_joints)
            self.write(joint_goal, duration_scale)
            assert len(joint_goal) == 7
    
    def reset(self, goal_joints):
        if self.arm_enabled:
            assert len(goal_joints) == 7 # 7 DOF joints 

            ### work directly with joints input
            cur_joints = self.joint_reader.get_most_recent_msg()
            delay_scale = np.linalg.norm(cur_joints - goal_joints)
            # assert delay_scale < 4 # official-version
            assert delay_scale < 6 # workable-version
            self.write(goal_joints, delay_scale*3) # [absolute control]
            rospy.sleep(delay_scale*4) # waiting for execution

            # ### work with pose input -> TODO: not workable due to the unsolvable situation from Inverse Kinematics
            # cur_joints = self.joint_reader.get_most_recent_msg()
            # goal_joints, duration_scale = self.find_ik(pose[:3], pose[3:7])
            # if goal_joints is not None:
            #     delay_scale = np.linalg.norm(cur_joints - goal_joints)
            #     self.write(goal_joints, delay_scale*3) # [absolute control]


"""
==========================================================================
====== TODO: utilize internal_state update to eliminate deviation ========
== (not working currently due to the inaccurate internal estimation) =====
==========================================================================
"""

    # def update_internal_state(self, pos_delta, euler_delta):
    #     self._internal_state_pose = np.array(self._internal_state_pose) + np.array(pos_delta)
    #     quat_delta = euler_to_quat(euler_delta)
    #     self._internal_state_quat = add_quats(quat_delta, self._internal_state_quat)
    
    # def process_action(self, action):
    #     # convert deltas to absolute positions
    #     pos_delta, euler_delta = action[:3], action[3:6]
    #     cur_pos, cur_quat = self.arm_reader.get_transform(target_link=f'/arm_{self.side}_tool_link', base_link='/torso_lift_link')
    #     # compensate for robot's pose deviation
    #     robot_pos_offset = np.array(cur_pos) - np.array(self._internal_state_pose)
    #     pos_delta_exe = np.array(pos_delta) - np.array(robot_pos_offset)
    #     # compensate for robot's quat deviation
    #     robot_quat_offset = quat_diff(cur_quat, self._internal_state_quat)
    #     quat_delta = euler_to_quat(euler_delta)
    #     quat_delta_exe = quat_diff(quat_delta, robot_quat_offset)
    #     # add executive actions
    #     target_pos = np.array(pos_delta_exe) + np.array(cur_pos)
    #     target_quat = add_quats(quat_delta_exe, cur_quat)
    #     # update internal state
    #     self.update_internal_state(pos_delta, euler_delta)
    #     # return values
    #     return target_pos, target_quat

    # def reset(self, goal_joints):
    #     if self.arm_enabled:
    #         assert len(goal_joints) == 7 # 7 DOF joints 
    #         cur_joints = self.joint_reader.get_most_recent_msg()
    #         delay_scale = np.linalg.norm(cur_joints - goal_joints)
    #         assert delay_scale < 6 # workable-version
    #         self.write(goal_joints, delay_scale*3) # [absolute control]
    #         rospy.sleep(delay_scale*5) # waiting for execution
    #         self._internal_state_pose, self._internal_state_quat = \
    #           self.arm_reader.get_transform(target_link=f'/arm_{self.side}_tool_link', base_link='/torso_lift_link') # assign internal states
    
