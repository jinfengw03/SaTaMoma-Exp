#!/usr/bin/env python

import rospy
import numpy as np
import jax.numpy as jnp
import os
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from manipulator import Manipulator  # 假设自定义模块
from oscbf_configs import OSCBFVelocityConfig  # 假设自定义模块
from cbfpy import CBF  # 假设已安装
import tf  # ROS1 TF
from geometry_msgs.msg import PointStamped, Point

# 设置 NumPy 数组打印格式
np.set_printoptions(precision=3, suppress=True)

class CollisionsVelocityConfig(OSCBFVelocityConfig):
    def __init__(
        self,
        robot: Manipulator,
        collision_positions: np.ndarray = np.array([]),
        collision_radii: np.ndarray = np.array([]),
    ):
        self.collision_positions = jnp.atleast_2d(collision_positions)
        self.collision_radii = jnp.ravel(collision_radii)
        super(CollisionsVelocityConfig, self).__init__(robot)  # ROS1 语法

    def h_1(self, z, **kwargs):
        q = z[: self.num_joints]
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        h_collision = jnp.array([1.0])  # 默认值
        if self.collision_positions.size > 0:
            center_deltas = (
                robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
            ).reshape(-1, 3)
            radii_sums = (
                robot_collision_radii[:, None] + self.collision_radii[None, :]
            ).reshape(-1)
            h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums
        return h_collision

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2

class SafetyFilterNode:
    def __init__(self):
        rospy.init_node('safety_filter_right_node')

        self.motion_blocked = False  # 新增：危险后阻止动作
        self.command_sent = False  # 是否已发送命令（只发送一次）- 对应 tried_once
        self.dt = 0.2  # 将 dt 移到前面，因为后面会用到  
        
        # 设置最大加速度限制（rad/s^2），避免剧烈抖动
        self.max_acceleration = 2.0  # 每个关节的最大加速度
        self.last_velocity = np.zeros(7)  # 记录上一次的速度
        
        # 初始化标志：是否已到达初始位置
        self.reached_initial_position = False
        self.position_tolerance = 0.05  # 位置容差（rad）

        self.arm_right_link_names = [
            'arm_right_1_link',  # sphere_right_1
            'arm_right_2_link',  # sphere_right_2
            'arm_right_3_link',  # sphere_right_3, sphere_right_9, sphere_right_10
            'arm_right_4_link',  # sphere_right_4, sphere_right_8
            'arm_right_5_link',  # sphere_right_5, sphere_right_12
            'arm_right_6_link',  # sphere_right_6, sphere_right_11
            'arm_right_7_link',  # sphere_right_7
        ]
        self.sphere_offsets = [
            [(0.0, 0.0, 0.0)],                                 # arm_right_1_link
            [(0.0, 0.0, 0.0)],                                 # arm_right_2_link
            [(0.0, 0.0, 0.0), (0.0, 0.0, -0.08), (0.0, 0.0, -0.16)],  # arm_right_3_link
            [(0.0, 0.01, 0.02), (-0.08, 0.02, 0.01)],          # arm_right_4_link
            [(0.0, 0.0, 0.02), (0.0, 0.0, 0.08)],              # arm_right_5_link
            [(0.09, 0.0, 0.0), (0.15, 0.0, 0.0)],              # arm_right_6_link
            [(0.0, 0.0, 0.0)],                                 # arm_right_7_link
        ]
        self.sphere_radii = [
            0.08,      # sphere_right_1
            0.07,      # sphere_right_2
            0.07, 0.07, 0.07,  # sphere_right_3, sphere_right_9, sphere_right_10
            0.08, 0.07,        # sphere_right_4, sphere_right_8
            0.07, 0.07,        # sphere_right_5, sphere_right_12
            0.07, 0.07,        # sphere_right_6, sphere_right_11
            0.07               # sphere_right_7
        ]
        self.root_frame = 'torso_lift_link'  # URDF 根坐标系
        self.tf_listener = tf.TransformListener()  # ROS1 TF Listener
        # 定义碰撞数据（根据你的xacro球体配置）
        link_1_pos = ((0.0, 0.0, 0.0),)  # sphere_right_1
        link_1_radii = (0.08,)

        link_2_pos = ((0.0, 0.0, 0.0),)  # sphere_right_2
        link_2_radii = (0.07,)

        link_3_pos = (
            (0.0, 0.0, 0.0),    # sphere_right_3
            (0.0, 0.0, -0.08),  # sphere_right_9
            (0.0, 0.0, -0.16),  # sphere_right_10
        )
        link_3_radii = (0.07, 0.07, 0.07)

        link_4_pos = (
            (0.0, 0.01, 0.02),   # sphere_right_4
            (-0.08, 0.02, 0.01), # sphere_right_8
        )
        link_4_radii = (0.08, 0.07)

        link_5_pos = (
            (0.0, 0.0, 0.02),    # sphere_right_5
            (0.0, 0.0, 0.08),    # sphere_right_12
        )
        link_5_radii = (0.07, 0.07)

        link_6_pos = (
            (0.09, 0.0, 0.0),    # sphere_right_6
            (0.15, 0.0, 0.0),    # sphere_right_11
        )
        link_6_radii = (0.07, 0.07)

        link_7_pos = ((0.0, 0.0, 0.0),)  # sphere_right_7
        link_7_radii = (0.07,)

        positions_list = (
            link_1_pos, link_2_pos, link_3_pos, link_4_pos,
            link_5_pos, link_6_pos, link_7_pos,
        )
        radii_list = (
            link_1_radii, link_2_radii, link_3_radii, link_4_radii,
            link_5_radii, link_6_radii, link_7_radii,
        )
        collision_data_right = {"positions": positions_list, "radii": radii_list}
        # 加载右臂 URDF
        urdf_path = '/home/rhino/tiago_dual_public_ws/src/tiago_safety/urdf/tiago_right_arm.urdf'
        self.right_arm = Manipulator.from_urdf(
            urdf_filename=urdf_path,
            ee_offset=np.eye(4),
            collision_data=collision_data_right
        )
        # 初始化 CBF
        self.cbf_right = CBF.from_config(CollisionsVelocityConfig(
            robot=self.right_arm,
            collision_positions=np.array([]),
            collision_radii=np.array([])
        ))

        self.target_q_right = np.array([1.0, 0.5, 0.0, 1.3, 0.0, 0.0, 0.0])
        self.pos_sub_right = rospy.Subscriber('/custom_arm_pos_right', Float64MultiArray, self.pos_callback_right, queue_size=10)

        self.arm_right_joint_names = [
            'arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint',
            'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint',
            'arm_right_7_joint'
        ]
        self.current_q_right = np.zeros(7)
        self.u_nom_right = np.zeros(7)
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback, queue_size=10)
        self.sphere_sub = rospy.Subscriber('/detected_spheres', Float64MultiArray, self.sphere_callback, queue_size=10)
        self.original_obstacles = []  # 原始障碍物（base_footprint 帧，用于日志）
        self.transformed_obstacles = []  # 变换后障碍物（torso_lift_link 帧，用于 CBF）
        self.arm_pub_right = rospy.Publisher('/arm_right_controller/command', JointTrajectory, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.update)  # ROS1 Timer
        self.tf_listener = tf.TransformListener(rospy.Duration(10.0))  # 缓存 10 秒
        rospy.loginfo('Safety Filter Node started')
        
        # 启动时直接移动到初始位置，不考虑安全过滤
        rospy.sleep(1.0)  # 等待发布者初始化
        self.send_initial_position()

    def send_initial_position(self):
        """启动时直接发送初始位置到机械臂控制器，不经过安全过滤"""
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = self.arm_right_joint_names
        point = JointTrajectoryPoint()
        point.positions = self.target_q_right.tolist()
        point.velocities = [0.0] * 7
        point.accelerations = [0.0] * 7
        point.time_from_start = rospy.Duration(3.0)  # 3秒到达初始位置
        traj.points = [point]
        self.arm_pub_right.publish(traj)
        rospy.loginfo(f'Sent initial position directly to arm: {self.target_q_right}')

    def limit_acceleration(self, desired_velocity):
        """限制关节加速度，避免剧烈抖动
        
        Args:
            desired_velocity: 期望的速度 (rad/s)
        
        Returns:
            限制加速度后的速度 (rad/s)
        """
        # 计算加速度 = (新速度 - 旧速度) / dt
        acceleration = (desired_velocity - self.last_velocity) / self.dt
        
        # 限制加速度幅值
        acceleration_magnitude = np.abs(acceleration)
        acceleration_limited = np.where(
            acceleration_magnitude > self.max_acceleration,
            np.sign(acceleration) * self.max_acceleration,
            acceleration
        )
        
        # 计算限制后的速度
        limited_velocity = self.last_velocity + acceleration_limited * self.dt
        
        # 更新记录的速度
        self.last_velocity = limited_velocity.copy()
        
        return limited_velocity

    def get_collision_spheres_torso_lift_link(self):
        """获取机械臂所有球体在 torso_lift_link 坐标系下的位置和半径"""
        positions = []
        radii = []
        for link_name, offsets in zip(self.arm_right_link_names, self.sphere_offsets):
            for offset in offsets:
                pt = PointStamped()
                pt.header.frame_id = link_name
                pt.header.stamp = rospy.Time(0)  # 使用最新可用的变换
                pt.point.x, pt.point.y, pt.point.z = offset
                try:
                    pt_torso = self.tf_listener.transformPoint('torso_lift_link', pt)  # ROS1 transformPoint
                    positions.append([pt_torso.point.x, pt_torso.point.y, pt_torso.point.z])
                    radii.append(self.sphere_radii[len(radii)])
                except Exception as e:
                    rospy.logwarn(f'TF变换失败: {e}')
        return np.array(positions), np.array(radii)

    def pos_callback_right(self, msg):
        self.target_q_right = np.array(msg.data)
        rospy.loginfo(f'New target received: {self.target_q_right}')
        # 重置标志，允许发送新命令
        self.command_sent = False
        self.motion_blocked = False

    def joint_callback(self, msg):
        try:
            idx = [msg.name.index(j) for j in self.arm_right_joint_names]
            self.current_q_right = np.array([msg.position[i] for i in idx])
            # 减少日志输出频率，避免影响性能
            # rospy.loginfo(f'Current joint angles: {self.current_q_right}')
        except ValueError:
            rospy.logwarn('Joint names not found in /joint_states')

    def sphere_callback(self, msg):
        flat_data = msg.data
        self.original_obstacles = []
        self.transformed_obstacles = []
        
        # 接收的球体已经在 torso_lift_link 坐标系下
        # 直接使用，不再进行变换
        for i in range(0, len(flat_data), 4):
            if i + 3 < len(flat_data):
                x, y, z, r = flat_data[i:i+4]
                # 直接使用（已经在 torso_lift_link 坐标系下）
                self.original_obstacles.append([x, y, z, r])
                self.transformed_obstacles.append([x, y, z, r])
                rospy.loginfo_throttle(2.0, f'接收障碍物 ({self.root_frame} 坐标系): ({x:.2f}, {y:.2f}, {z:.2f}), r={r:.2f}')

        transformed_array = np.array(self.transformed_obstacles) if self.transformed_obstacles else np.array([])
        self.cbf_right = CBF.from_config(CollisionsVelocityConfig(
            robot=self.right_arm,
            collision_positions=transformed_array[:, :3] if transformed_array.size else np.array([]),
            collision_radii=transformed_array[:, 3] if transformed_array.size else np.array([])
        ))
        rospy.loginfo_throttle(2.0, f'CBF updated with {len(self.transformed_obstacles)} obstacles (torso_lift_link 坐标系)')

    def update(self, event):  # ROS1 Timer callback
        # 检查是否已到达初始位置
        if not self.reached_initial_position:
            position_error = np.abs(self.current_q_right - self.target_q_right)
            if np.all(position_error < self.position_tolerance):
                self.reached_initial_position = True
                rospy.loginfo('Reached initial position! Safety filter is now active.')
            else:
                rospy.loginfo_throttle(2.0, f'Moving to initial position... error: {np.max(position_error):.3f} rad')
                return  # 不启动安全过滤器，让机械臂自由移动到初始位置
        
        # 获取机械臂所有球体在 torso_lift_link 坐标系下的位置和半径
        robot_collision_positions, robot_collision_radii = self.get_collision_spheres_torso_lift_link()

        # 障碍物坐标和半径（已经在 torso_lift_link 坐标系）
        if self.original_obstacles:
            obstacle_positions = np.array([obs[:3] for obs in self.original_obstacles])
            obstacle_radii = np.array([obs[3] for obs in self.original_obstacles])
        else:
            obstacle_positions = np.empty((0, 3))
            obstacle_radii = np.empty((0,))

        # 计算碰撞距离（与 CBF h_1 逻辑一致）
        if obstacle_positions.size > 0 and robot_collision_positions.size > 0:
            center_deltas = (
                robot_collision_positions[:, None, :] - obstacle_positions[None, :, :]
            ).reshape(-1, 3)
            radii_sums = (
                robot_collision_radii[:, None] + obstacle_radii[None, :]
            ).reshape(-1)
            h_collision = np.linalg.norm(center_deltas, axis=1) - radii_sums
            
            # 输出距离最近障碍物的距离
            min_distance = np.min(h_collision)
            min_index = np.argmin(h_collision)
            rospy.loginfo_throttle(2.0, f'最近障碍物距离: {min_distance:.4f} m (索引: {min_index})')
            
            if np.any(h_collision < 0.2):
                rospy.logwarn('WARNING: Robot is very close to obstacle!')
            if np.any(h_collision < 0.03):
                rospy.logwarn('ERROR: COLLISION!')
        else:
            h_collision = np.array([1.0])
            rospy.loginfo_throttle(2.0, '没有检测到障碍物')

        # 判断是否有距离小于等于0.87米
        cbf_enabled = np.any(h_collision <= 0.87) if obstacle_positions.size > 0 and robot_collision_positions.size > 0 else False

        # 只尝试一次机制
        if self.motion_blocked:
            rospy.logwarn('Motion blocked due to previous safety violation.')
            return

        if cbf_enabled:
            if self.command_sent:
                rospy.loginfo('Already tried once, not sending further commands.')
                return
            self.command_sent = True  # 标记已尝试

            z = self.current_q_right
            u_nom_right = (self.target_q_right - self.current_q_right) / self.dt
            u_safe = self.cbf_right.safety_filter(z, u_nom_right)
            
            # 应用加速度限制，避免剧烈抖动
            u_safe_limited = self.limit_acceleration(u_safe)
            new_pos = self.current_q_right + u_safe_limited * self.dt

            # 如果检测到碰撞风险，阻止后续动作
            if np.any(h_collision < 0):
                rospy.logerr('Collision risk detected, blocking further motion!')
                self.motion_blocked = True
                return

            traj = JointTrajectory()
            traj.joint_names = self.arm_right_joint_names
            point = JointTrajectoryPoint()
            point.positions = new_pos.tolist()
            point.velocities = u_safe_limited.tolist()
            point.time_from_start = rospy.Duration(self.dt)
            traj.points = [point]
            self.arm_pub_right.publish(traj)
            rospy.loginfo(f'Safe velocity (u_safe): {u_safe_limited}')
            rospy.loginfo(f'CBF constraint h_1: {h_collision}')
        else:
            # 距离大于0.87米，直接发送目标位置，不做CBF过滤
            if self.command_sent:
                rospy.loginfo('Already tried once, not sending further commands.')
                return
            self.command_sent = True  # 标记已尝试

            # 计算期望速度并应用加速度限制
            desired_velocity = (self.target_q_right - self.current_q_right) / self.dt
            velocity_limited = self.limit_acceleration(desired_velocity)
            
            traj = JointTrajectory()
            traj.joint_names = self.arm_right_joint_names
            point = JointTrajectoryPoint()
            point.positions = self.target_q_right.tolist()
            point.velocities = velocity_limited.tolist()
            point.time_from_start = rospy.Duration(self.dt)
            traj.points = [point]
            self.arm_pub_right.publish(traj)
            rospy.loginfo('CBF not enabled, direct target position sent.')

def main():
    node = SafetyFilterNode()
    rospy.spin()  # ROS1 spin

if __name__ == '__main__':
    main()