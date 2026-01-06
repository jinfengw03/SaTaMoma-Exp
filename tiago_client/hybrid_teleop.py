#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import copy
import sys
import tty
import termios
import numpy as np
import tf
from geometry_msgs.msg import PoseStamped, Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from sensor_msgs.msg import JointState

class HybridTeleop:
    def __init__(self):
        rospy.init_node('hybrid_teleop')

        # TF Listener
        self.tf_listener = tf.TransformListener()
        rospy.sleep(1.0)

        # MoveIt IK Service
        rospy.loginfo("Waiting for /compute_ik service...")
        try:
            rospy.wait_for_service('/compute_ik', timeout=5.0)
            self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
            rospy.loginfo("/compute_ik service connected")
        except rospy.ROSException:
            rospy.logerr("IK Service not available. Cartesian mode will not work.")

        # Publishers
        self.vel_pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)
        self.arm_right_pub = rospy.Publisher('/arm_right_controller/command', JointTrajectory, queue_size=10)
        self.torso_pub = rospy.Publisher('/torso_controller/command', JointTrajectory, queue_size=10)
        self.head_pub = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=10)
        self.gripper_right_pub = rospy.Publisher('/gripper_right_controller/command', JointTrajectory, queue_size=10)

        # Joint Names
        self.arm_right_joint_names = [
            'arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint',
            'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint',
            'arm_right_7_joint'
        ]
        self.torso_joint_names = ['torso_lift_joint']
        self.head_joint_names = ['head_1_joint', 'head_2_joint']
        self.gripper_right_joint_names = ['gripper_right_left_finger_joint', 'gripper_right_right_finger_joint']

        # Limits
        self.arm_right_limits = [
            (-2.3, 2.3), (-2.2, 2.2), (-3.0, 3.0),
            (-3.0, 3.0), (-2.0, 2.0), (-2.1, 2.1), (-3.0, 3.0)
        ]
        self.torso_limits = (0.0, 0.35)
        self.head_1_limits = (-1.3, 1.3)
        self.head_2_limits = (-1.05, 0.785)

        # State Variables
        self.current_arm_right_joints = None
        self.current_torso_position = None
        self.head_positions = None
        
        # Control Mode
        self.mode = 'JOINT' # 'JOINT' or 'CARTESIAN'
        
        # Control Parameters
        self.cartesian_step = 0.02
        self.joint_step = 0.05
        self.target_pose_base = None # [x, y, z, qx, qy, qz, qw]
        self.target_arm_joints = None # List of 7 floats

        # Subscribers
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        self.torso_sub = rospy.Subscriber('/torso_controller/state', JointTrajectoryControllerState, self.torso_callback)

        self.settings = termios.tcgetattr(sys.stdin)

        # Wait for initial state
        rospy.loginfo("Waiting for joint states...")
        while not rospy.is_shutdown() and (self.current_arm_right_joints is None or self.current_torso_position is None or self.head_positions is None):
            rospy.sleep(0.1)
        rospy.loginfo("Joint states received.")

        # Move to initial safe pose
        self.move_to_initial_pose()
        
        # Initialize targets
        self.target_arm_joints = list(self.current_arm_right_joints)
        self.initialize_cartesian_target()

        self.print_usage()
        self.run()

    def print_usage(self):
        print("\n" + "="*50)
        print("TIAGo Hybrid Teleop (Right Arm Only)")
        print("="*50)
        print(f"Current Mode: [{self.mode}] (Press TAB to switch)")
        print("-" * 30)
        print("Common Controls:")
        print("  Base:    WASD (Move) / QE (Rotate) / Space (Stop)")
        print("  Torso:   M (Up) / N (Down)")
        print("  Head:    Arrow Keys (Left/Right/Up/Down)")
        print("  Gripper: P (Open) / ; (Close)")
        print("  Reset:   B (Back to Initial Pose)")
        print("  Exit:    Ctrl+C")
        print("-" * 30)
        if self.mode == 'CARTESIAN':
            print("Cartesian Mode (Base Frame):")
            print("  I / K :  X +/- (Forward/Backward)")
            print("  J / L :  Y +/- (Left/Right)")
            print("  U / O :  Z +/- (Up/Down)")
            print("  R     :  Reset Target to Current")
        else:
            print("Joint Mode (Right Arm):")
            print("  R / F :  J1 +/-")
            print("  T / G :  J2 +/-")
            print("  Y / H :  J3 +/-")
            print("  U / J :  J4 +/-")
            print("  I / K :  J5 +/-")
            print("  O / L :  J6 +/-")
            print("  Z / X :  J7 +/-")
        print("="*50)

    def joint_callback(self, msg):
        try:
            # Update arm joints
            temp_joints = []
            for name in self.arm_right_joint_names:
                if name in msg.name:
                    idx = msg.name.index(name)
                    temp_joints.append(msg.position[idx])
            if len(temp_joints) == 7:
                self.current_arm_right_joints = temp_joints
            
            # Update head joints (init)
            if self.head_positions is None:
                temp_head = []
                for name in self.head_joint_names:
                    if name in msg.name:
                        idx = msg.name.index(name)
                        temp_head.append(msg.position[idx])
                if len(temp_head) == 2:
                    self.head_positions = temp_head
        except Exception:
            pass

    def torso_callback(self, msg):
        try:
            if 'torso_lift_joint' in msg.joint_names:
                idx = msg.joint_names.index('torso_lift_joint')
                self.current_torso_position = msg.actual.positions[idx]
        except Exception:
            pass

    def move_to_initial_pose(self):
        rospy.loginfo("Moving to initial safe pose...")
        
        # Wait for publishers to connect
        while (self.arm_right_pub.get_num_connections() == 0 or self.torso_pub.get_num_connections() == 0) and not rospy.is_shutdown():
            rospy.loginfo("Waiting for controller subscribers...")
            rospy.sleep(0.5)

        # Arm
        initial_arm_pose = [0.6999653135630082, 0, 1.1521369638820005, 0.9994929781529036, -0.14998109119333058, -0.003483463594645464, 1.9663696343208414e-05]
        self.publish_arm_joints(initial_arm_pose, duration=3.0)
        
        # Torso
        initial_torso_pos = 0.24
        self.publish_torso(initial_torso_pos, duration=3.0)

        rospy.sleep(3.5)
        rospy.loginfo("Initial pose reached.")
        
        # Sync internal state
        self.target_arm_joints = list(initial_arm_pose)
        self.current_torso_position = initial_torso_pos
        self.initialize_cartesian_target()

    def initialize_cartesian_target(self):
        try:
            self.tf_listener.waitForTransform('base_footprint', 'arm_right_tool_link', rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform('base_footprint', 'arm_right_tool_link', rospy.Time(0))
            self.target_pose_base = trans + rot
            rospy.loginfo(f"Cartesian Target Initialized: {self.target_pose_base[:3]}")
        except Exception as e:
            rospy.logwarn(f"Failed to init Cartesian target: {e}")

    def solve_ik(self, target_pose_base):
        try:
            ps_base = PoseStamped()
            ps_base.header.frame_id = 'base_footprint'
            ps_base.header.stamp = rospy.Time(0)
            ps_base.pose.position.x = target_pose_base[0]
            ps_base.pose.position.y = target_pose_base[1]
            ps_base.pose.position.z = target_pose_base[2]
            ps_base.pose.orientation.x = target_pose_base[3]
            ps_base.pose.orientation.y = target_pose_base[4]
            ps_base.pose.orientation.z = target_pose_base[5]
            ps_base.pose.orientation.w = target_pose_base[6]

            self.tf_listener.waitForTransform('torso_lift_link', 'base_footprint', rospy.Time(0), rospy.Duration(0.1))
            ps_torso = self.tf_listener.transformPose('torso_lift_link', ps_base)

            req = GetPositionIKRequest()
            req.ik_request.group_name = "arm_right"
            req.ik_request.robot_state.joint_state.name = self.arm_right_joint_names
            req.ik_request.robot_state.joint_state.position = self.current_arm_right_joints
            req.ik_request.avoid_collisions = True
            req.ik_request.pose_stamped = ps_torso
            req.ik_request.timeout = rospy.Duration(0.1)

            res = self.ik_service(req)

            if res.error_code.val == res.error_code.SUCCESS:
                sol_joints = res.solution.joint_state
                output_joints = []
                for name in self.arm_right_joint_names:
                    if name in sol_joints.name:
                        idx = sol_joints.name.index(name)
                        output_joints.append(sol_joints.position[idx])
                return output_joints
            else:
                return None
        except Exception:
            return None

    def publish_arm_joints(self, joints, duration=0.5):
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = self.arm_right_joint_names
        point = JointTrajectoryPoint()
        point.positions = joints
        point.velocities = [0.0] * 7
        point.accelerations = [0.0] * 7
        point.time_from_start = rospy.Duration(duration)
        traj.points.append(point)
        self.arm_right_pub.publish(traj)

    def publish_torso(self, position, duration=0.5):
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = self.torso_joint_names
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = rospy.Duration(duration)
        traj.points.append(point)
        self.torso_pub.publish(traj)

    def publish_head(self):
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = self.head_joint_names
        point = JointTrajectoryPoint()
        point.positions = copy.deepcopy(self.head_positions)
        point.time_from_start = rospy.Duration(0.5)
        traj.points.append(point)
        self.head_pub.publish(traj)

    def publish_gripper(self, position):
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = self.gripper_right_joint_names
        point = JointTrajectoryPoint()
        point.positions = [position, position]
        point.time_from_start = rospy.Duration(1.0)
        traj.points.append(point)
        self.gripper_right_pub.publish(traj)
        rospy.loginfo(f"Gripper: {position}")

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
        if key == '\x1b': # Handle arrow keys
            key += sys.stdin.read(2)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run(self):
        try:
            rate = rospy.Rate(10)
            while not rospy.is_shutdown():
                key = self.get_key()
                twist = Twist()
                
                # Exit
                if key == '\x03':
                    break
                
                # Mode Switch
                if key == '\t':
                    if self.mode == 'JOINT':
                        self.mode = 'CARTESIAN'
                        self.initialize_cartesian_target()
                    else:
                        self.mode = 'JOINT'
                        # Sync joint target to current
                        self.target_arm_joints = list(self.current_arm_right_joints)
                    self.print_usage()
                    continue

                # Base Control (Always Active)
                if key == 'w': twist.linear.x = 0.5
                elif key == 's': twist.linear.x = -0.5
                elif key == 'a': twist.angular.z = 1.0
                elif key == 'd': twist.angular.z = -1.0
                elif key == 'q': twist.angular.z = 1.5
                elif key == 'e': twist.angular.z = -1.5
                elif key == ' ': 
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                
                if key in ['w', 's', 'a', 'd', 'q', 'e', ' ']:
                    self.vel_pub.publish(twist)

                # Common Controls
                # Torso
                if key == 'm':
                    self.current_torso_position = min(self.current_torso_position + 0.02, self.torso_limits[1])
                    self.publish_torso(self.current_torso_position)
                elif key == 'n':
                    self.current_torso_position = max(self.current_torso_position - 0.02, self.torso_limits[0])
                    self.publish_torso(self.current_torso_position)
                
                # Gripper
                elif key == 'p': self.publish_gripper(0.05) # Open
                elif key == ';': self.publish_gripper(0.0)  # Close

                # Reset
                elif key == 'b':
                    self.move_to_initial_pose()

                # Head (Arrow Keys)
                elif key == '\x1b[A': # Up
                    self.head_positions[1] = max(self.head_positions[1] - 0.1, self.head_2_limits[0])
                    self.publish_head()
                elif key == '\x1b[B': # Down
                    self.head_positions[1] = min(self.head_positions[1] + 0.1, self.head_2_limits[1])
                    self.publish_head()
                elif key == '\x1b[C': # Right
                    self.head_positions[0] = max(self.head_positions[0] - 0.1, self.head_1_limits[0])
                    self.publish_head()
                elif key == '\x1b[D': # Left
                    self.head_positions[0] = min(self.head_positions[0] + 0.1, self.head_1_limits[1])
                    self.publish_head()

                # Mode Specific
                if self.mode == 'CARTESIAN':
                    if self.target_pose_base is not None:
                        updated = False
                        if key == 'i': 
                            self.target_pose_base[0] += self.cartesian_step; updated = True
                        elif key == 'k': 
                            self.target_pose_base[0] -= self.cartesian_step; updated = True
                        elif key == 'j': 
                            self.target_pose_base[1] += self.cartesian_step; updated = True
                        elif key == 'l': 
                            self.target_pose_base[1] -= self.cartesian_step; updated = True
                        elif key == 'u': 
                            self.target_pose_base[2] += self.cartesian_step; updated = True
                        elif key == 'o': 
                            self.target_pose_base[2] -= self.cartesian_step; updated = True
                        elif key == 'r':
                            self.initialize_cartesian_target()
                            rospy.loginfo("Reset Target")

                        if updated:
                            sol = self.solve_ik(self.target_pose_base)
                            if sol:
                                self.publish_arm_joints(sol)
                                self.target_arm_joints = list(sol) # Keep joint target synced
                            else:
                                # Revert
                                if key == 'i': self.target_pose_base[0] -= self.cartesian_step
                                elif key == 'k': self.target_pose_base[0] += self.cartesian_step
                                elif key == 'j': self.target_pose_base[1] -= self.cartesian_step
                                elif key == 'l': self.target_pose_base[1] += self.cartesian_step
                                elif key == 'u': self.target_pose_base[2] -= self.cartesian_step
                                elif key == 'o': self.target_pose_base[2] += self.cartesian_step
                                rospy.logwarn("IK Failed")

                elif self.mode == 'JOINT':
                    updated = False
                    # J1
                    if key == 'r': self.target_arm_joints[0] += self.joint_step; updated = True
                    elif key == 'f': self.target_arm_joints[0] -= self.joint_step; updated = True
                    # J2
                    elif key == 't': self.target_arm_joints[1] += self.joint_step; updated = True
                    elif key == 'g': self.target_arm_joints[1] -= self.joint_step; updated = True
                    # J3
                    elif key == 'y': self.target_arm_joints[2] += self.joint_step; updated = True
                    elif key == 'h': self.target_arm_joints[2] -= self.joint_step; updated = True
                    # J4
                    elif key == 'u': self.target_arm_joints[3] += self.joint_step; updated = True
                    elif key == 'j': self.target_arm_joints[3] -= self.joint_step; updated = True
                    # J5
                    elif key == 'i': self.target_arm_joints[4] += self.joint_step; updated = True
                    elif key == 'k': self.target_arm_joints[4] -= self.joint_step; updated = True
                    # J6
                    elif key == 'o': self.target_arm_joints[5] += self.joint_step; updated = True
                    elif key == 'l': self.target_arm_joints[5] -= self.joint_step; updated = True
                    # J7
                    elif key == 'z': self.target_arm_joints[6] += self.joint_step; updated = True
                    elif key == 'x': self.target_arm_joints[6] -= self.joint_step; updated = True

                    if updated:
                        # Clamp limits
                        for i in range(7):
                            self.target_arm_joints[i] = max(min(self.target_arm_joints[i], self.arm_right_limits[i][1]), self.arm_right_limits[i][0])
                        self.publish_arm_joints(self.target_arm_joints)

                rate.sleep()

        except rospy.ROSInterruptException:
            pass
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

if __name__ == '__main__':
    HybridTeleop()
