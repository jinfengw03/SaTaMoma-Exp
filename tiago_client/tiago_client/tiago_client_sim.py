import os
import rospy
import numpy as np
import tf
from collections import OrderedDict
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Twist
from tiago_client.oculus_teleop.teleop_policy import TeleopPolicy
from tiago_client.oculus_teleop.teleop_core import TeleopObservation
from tiago_client.utils.ik_solver import TiagoIK
from tiago_client.tiago_safety.safety_filter_right import JointSafetyFilter
from tiago_client.utils.transformations import quat_to_euler, euler_to_quat, add_angles

class TiagoClientSim:
    """
    Simulation client for TIAGO that interacts directly with ROS/Gazebo.
    Replaces the HTTP communication of TiagoClient with ROS publishers/subscribers.
    """
    def __init__(self, use_teleop=True):
        rospy.init_node('tiago_client_sim', anonymous=True)
        
        self.tf_listener = tf.TransformListener()
        
        # Joint names
        self.arm_right_names = [f'arm_right_{i}_joint' for i in range(1, 8)]
        self.arm_left_names = [f'arm_left_{i}_joint' for i in range(1, 8)]
        self.torso_names = ['torso_lift_joint']
        self.gripper_right_names = ['gripper_right_left_finger_joint', 'gripper_right_right_finger_joint']
        self.gripper_left_names = ['gripper_left_left_finger_joint', 'gripper_left_right_finger_joint']
        
        # Current state
        self.current_joints = {}
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self._joint_callback)
        
        # Publishers
        self.arm_right_pub = rospy.Publisher('/arm_right_controller/command', JointTrajectory, queue_size=10)
        self.arm_left_pub = rospy.Publisher('/arm_left_controller/command', JointTrajectory, queue_size=10)
        self.torso_pub = rospy.Publisher('/torso_controller/command', JointTrajectory, queue_size=10)
        self.gripper_right_pub = rospy.Publisher('/parallel_gripper_right_controller/command', JointTrajectory, queue_size=10)
        self.gripper_left_pub = rospy.Publisher('/parallel_gripper_left_controller/command', JointTrajectory, queue_size=10)
        self.base_pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)
        
        # Initialize Teleop, IK, and Safety
        self.teleop = None
        if use_teleop:
            from tiago_client.oculus_teleop.configs.only_vr import teleop_config
            self.teleop = TeleopPolicy(teleop_config)
            self.teleop.start()
            
            self.ik_solvers = {
                'right': TiagoIK(side='right'),
                'left': TiagoIK(side='left')
            }

            urdf_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'urdf')
            urdf_right_arm_path = os.path.join(urdf_dir, 'tiago_right_arm.urdf')
            urdf_left_arm_path = os.path.join(urdf_dir, 'tiago_left_arm.urdf')
            self.safety_filters = {
                'right': JointSafetyFilter(urdf_right_arm_path, side='right'),
                'left': JointSafetyFilter(urdf_left_arm_path, side='left')
            }
            
        print("[TiagoClientSim] Initialized and connected to ROS topics.")

    def _joint_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = pos

    def get_state_wo_vis(self):
        """Retrieves the robot state from ROS (joints and TF poses)."""
        state = {}
        
        # Get joint positions
        state['right_joints'] = np.array([self.current_joints.get(n, 0.0) for n in self.arm_right_names])
        state['left_joints'] = np.array([self.current_joints.get(n, 0.0) for n in self.arm_left_names])
        state['torso'] = np.array([self.current_joints.get(n, 0.0) for n in self.torso_names])
        
        # Get Cartesian poses via TF
        for side in ['right', 'left']:
            try:
                # Pose of EE relative to torso_lift_link (matching TiagoClient behavior)
                target_frame = f'arm_{side}_7_link'
                source_frame = 'torso_lift_link'
                (trans, quat) = self.tf_listener.lookupTransform(source_frame, target_frame, rospy.Time(0))
                state[side] = np.array(list(trans) + list(quat))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                state[side] = np.zeros(7)
                
        # Base pose (simplified for sim, usually relative to odom)
        try:
            (trans, quat) = self.tf_listener.lookupTransform('odom', 'base_footprint', rospy.Time(0))
            state['base_pose'] = np.array(list(trans) + list(quat))
        except:
            state['base_pose'] = np.zeros(7)
            
        return state

    def step(self, action):
        """
        Publishes the action to ROS topics.
        :param action: Dictionary of actions.
        """
        dt = 0.1 # Command duration
        
        # 1. Arms
        for side, pub, names in [('right', self.arm_right_pub, self.arm_right_names), 
                                 ('left', self.arm_left_pub, self.arm_left_names)]:
            if side in action:
                traj = JointTrajectory()
                traj.joint_names = names
                point = JointTrajectoryPoint()
                point.positions = action[side][:7]
                point.time_from_start = rospy.Duration(dt)
                traj.points.append(point)
                pub.publish(traj)
                
                # Gripper (if present in action[side][7])
                gripper_pub = self.gripper_right_pub if side == 'right' else self.gripper_left_pub
                gripper_names = self.gripper_right_names if side == 'right' else self.gripper_left_names
                g_traj = JointTrajectory()
                g_traj.joint_names = gripper_names
                g_point = JointTrajectoryPoint()
                # TIAGO parallel gripper: action[side][7] is usually 0 (closed) to 1 (open)
                # Map to joint values (approx 0.0 to 0.045)
                val = action[side][7] * 0.045
                g_point.positions = [val, val]
                g_point.time_from_start = rospy.Duration(dt)
                g_traj.points.append(g_point)
                gripper_pub.publish(g_traj)

        # 2. Torso
        if 'torso' in action:
            traj = JointTrajectory()
            traj.joint_names = self.torso_names
            point = JointTrajectoryPoint()
            point.positions = [action['torso']] if np.isscalar(action['torso']) else action['torso']
            point.time_from_start = rospy.Duration(dt)
            traj.points.append(point)
            self.torso_pub.publish(traj)

        # 3. Base
        if 'base' in action:
            vel = Twist()
            vel.linear.x = action['base'][0]
            vel.angular.z = action['base'][1]
            self.base_pub.publish(vel)

        return self.get_state_wo_vis(), {}

    def get_teleop_action(self, is_filter=False, obstacles=None):
        """
        Same logic as TiagoClient but uses local ROS state.
        """
        if self.teleop is None:
            return None, {}
            
        state = self.get_state_wo_vis()
        
        obs = TeleopObservation(
            left=state.get('left'),
            right=state.get('right'),
            base=state.get('base_pose'),
            torso=state.get('torso')[0] if isinstance(state.get('torso'), (list, np.ndarray)) else state.get('torso')
        )
        
        raw_action = self.teleop.get_action(obs, is_filter=is_filter)
        buttons = raw_action.extra.get('buttons', {})
        
        safe_action = {}
        
        for side in ['right', 'left']:
            if side in raw_action and raw_action[side] is not None:
                cartesian_delta = raw_action[side][:6]
                gripper_val = raw_action[side][6]
                
                cur_pose = state.get(side)
                cur_pos, cur_quat = cur_pose[:3], cur_pose[3:7]
                
                pos_delta, euler_delta = cartesian_delta[:3], cartesian_delta[3:6]
                cur_euler = quat_to_euler(cur_quat)
                target_pos = cur_pos + pos_delta
                target_euler = add_angles(euler_delta, cur_euler)
                target_quat = euler_to_quat(target_euler)
                
                joints_curr = state.get(f'{side}_joints')
                if joints_curr is not None:
                    joint_goal = self.ik_solvers[side].find_ik(target_pos, target_quat, joints_curr)
                    
                    if joint_goal is not None:
                        if obstacles is not None:
                            self.safety_filters[side].update_obstacles(obstacles)
                        
                        joint_safe = self.safety_filters[side].filter(joints_curr, joint_goal)
                        safe_action[side] = np.concatenate([joint_safe, [gripper_val]])
                    else:
                        safe_action[side] = np.concatenate([joints_curr, [gripper_val]])
        
        if 'base' in raw_action:
            safe_action['base'] = raw_action['base']
        if 'torso' in raw_action:
            safe_action['torso'] = raw_action['torso']
            
        return safe_action, buttons
