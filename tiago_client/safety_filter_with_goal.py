#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import jax.numpy as jnp
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from manipulator import Manipulator
from oscbf_configs import OSCBFVelocityConfig
from cbfpy import CBF
import tf
from geometry_msgs.msg import PointStamped, Point
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

np.set_printoptions(precision=3, suppress=True)

class CollisionsVelocityConfig(OSCBFVelocityConfig):
    def __init__(self, robot: Manipulator, collision_positions=np.array([]), collision_radii=np.array([])):
        self.collision_positions = jnp.atleast_2d(collision_positions)
        self.collision_radii = jnp.ravel(collision_radii)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        q = z[: self.num_joints]
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        h_collision = jnp.array([1.0])
        if self.collision_positions.size > 0:
            center_deltas = (robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]).reshape(-1, 3)
            radii_sums = (robot_collision_radii[:, None] + self.collision_radii[None, :]).reshape(-1)
            h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums
        return h_collision

    def alpha(self, h):
        return 10.0 * h

class SafetyFilterWithGoal:
    def __init__(self):
        rospy.init_node('safety_filter_with_goal_node', anonymous=False)

        self.dt = 0.2
        self.max_acceleration = 1.7
        self.last_velocity = np.zeros(7)

        # 目标点（薯片位置，在 odom 中）
        self.target_goal_odom = np.array([0.702, 0.0, 0.93])
        self.target_goal = None
        self.tf_listener = tf.TransformListener(rospy.Duration(10.0))

        self.reached_initial_position = False
        self.position_tolerance = 0.05
        self.motion_blocked = False
        self.escape_retry_count = 0
        self.command_sent = False

        self.arm_right_joint_names = [
            'arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint',
            'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint', 'arm_right_7_joint'
        ]

        # ================== 完整 collision_data_right（关键！）==================
        link_1_pos = ((0.0, 0.0, 0.0),);                     link_1_radii = (0.08,)
        link_2_pos = ((0.0, 0.0, 0.0),);                     link_2_radii = (0.07,)
        link_3_pos = ((0.0, 0.0, 0.0), (0.0, 0.0, -0.08), (0.0, 0.0, -0.16)); link_3_radii = (0.07, 0.07, 0.07)
        link_4_pos = ((0.0, 0.01, 0.02), (-0.08, 0.02, 0.01));               link_4_radii = (0.08, 0.07)
        link_5_pos = ((0.0, 0.0, 0.02), (0.0, 0.0, 0.08));                  link_5_radii = (0.07, 0.07)
        link_6_pos = ((0.09, 0.0, 0.0), (0.15, 0.0, 0.0));                  link_6_radii = (0.07, 0.07)
        link_7_pos = ((0.0, 0.0, 0.0),);                                           link_7_radii = (0.07,)

        positions_list = (link_1_pos, link_2_pos, link_3_pos, link_4_pos, link_5_pos, link_6_pos, link_7_pos)
        radii_list     = (link_1_radii, link_2_radii, link_3_radii, link_4_radii, link_5_radii, link_6_radii, link_7_radii)

        collision_data_right = {"positions": positions_list, "radii": radii_list}
        # =====================================================================

        urdf_path = '/home/rhino/tiago_dual_public_ws/src/tiago_safety/urdf/tiago_right_arm.urdf'
        self.right_arm = Manipulator.from_urdf(
            urdf_filename=urdf_path,
            ee_offset=np.eye(4),
            collision_data=collision_data_right
        )

        self.cbf_right = CBF.from_config(CollisionsVelocityConfig(robot=self.right_arm))

        rospy.wait_for_service('/compute_ik', timeout=10.0)
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)

        self.target_q_right = np.array([1.0, 0.5, 0.0, 1.3, 0.0, 0.0, 0.0])
        self.current_q_right = np.zeros(7)
        self.original_obstacles = []

        rospy.Subscriber('/custom_arm_pos_right', Float64MultiArray, self.pos_callback_right, queue_size=10)
        rospy.Subscriber('/joint_states', JointState, self.joint_callback, queue_size=10)
        rospy.Subscriber('/detected_spheres', Float64MultiArray, self.sphere_callback, queue_size=10)

        self.arm_pub_right = rospy.Publisher('/arm_right_controller/command', JointTrajectory, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(self.dt), self.update)

        rospy.sleep(1.0)
        self.update_target_goal_transform()
        self.send_initial_position()
        rospy.loginfo("Safety Filter With Goal Node 启动完成！")

    def send_initial_position(self):
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = self.arm_right_joint_names
        point = JointTrajectoryPoint()
        point.positions = self.target_q_right.tolist()
        point.time_from_start = rospy.Duration(3.0)
        traj.points = [point]
        self.arm_pub_right.publish(traj)

    def update_target_goal_transform(self):
        try:
            p = PointStamped()
            p.header.frame_id = "odom"
            p.header.stamp = rospy.Time(0)
            p.point.x, p.point.y, p.point.z = self.target_goal_odom
            p_torso = self.tf_listener.transformPoint("torso_lift_link", p)
            self.target_goal = np.array([p_torso.point.x, p_torso.point.y, p_torso.point.z])
            rospy.loginfo_throttle(5.0, f"目标点更新: {self.target_goal}")
        except Exception as e:
            rospy.logwarn(f"TF变换失败，使用默认目标: {e}")
            self.target_goal = np.array([0.70, 0.0, 0.75])

    def limit_acceleration(self, v):
        a = (v - self.last_velocity) / self.dt
        a = np.clip(a, -self.max_acceleration, self.max_acceleration)
        v_lim = self.last_velocity + a * self.dt
        self.last_velocity = v_lim.copy()
        return v_lim

    def get_end_effector_position(self):
        try:
            p = PointStamped()
            p.header.frame_id = "arm_right_7_link"
            p.header.stamp = rospy.Time(0)
            p_torso = self.tf_listener.transformPoint("torso_lift_link", p)
            return np.array([p_torso.point.x, p_torso.point.y, p_torso.point.z])
        except:
            return None

    def get_current_quat(self):
        try:
            _, quat = self.tf_listener.lookupTransform("torso_lift_link", "arm_right_7_link", rospy.Time(0))
            return quat
        except:
            return [0, 0, 0, 1]

    def get_collision_spheres_torso_lift_link(self):
        """获取机械臂所有球体在 torso_lift_link 坐标系下的位置和半径（与 safety_filter_right.py 一致）"""
        arm_right_link_names = [
            'arm_right_1_link', 'arm_right_2_link', 'arm_right_3_link',
            'arm_right_4_link', 'arm_right_5_link', 'arm_right_6_link', 'arm_right_7_link'
        ]
        sphere_offsets = [
            [(0.0, 0.0, 0.0)],                                 # arm_right_1_link
            [(0.0, 0.0, 0.0)],                                 # arm_right_2_link
            [(0.0, 0.0, 0.0), (0.0, 0.0, -0.08), (0.0, 0.0, -0.16)],  # arm_right_3_link
            [(0.0, 0.01, 0.02), (-0.08, 0.02, 0.01)],          # arm_right_4_link
            [(0.0, 0.0, 0.02), (0.0, 0.0, 0.08)],              # arm_right_5_link
            [(0.09, 0.0, 0.0), (0.15, 0.0, 0.0)],              # arm_right_6_link
            [(0.0, 0.0, 0.0)],                                 # arm_right_7_link
        ]
        sphere_radii = [
            0.08,      # sphere_right_1
            0.07,      # sphere_right_2
            0.07, 0.07, 0.07,  # sphere_right_3, sphere_right_9, sphere_right_10
            0.08, 0.07,        # sphere_right_4, sphere_right_8
            0.07, 0.07,        # sphere_right_5, sphere_right_12
            0.07, 0.07,        # sphere_right_6, sphere_right_11
            0.07               # sphere_right_7
        ]
        
        positions = []
        radii = []
        for link_name, offsets in zip(arm_right_link_names, sphere_offsets):
            for offset in offsets:
                pt = PointStamped()
                pt.header.frame_id = link_name
                pt.header.stamp = rospy.Time(0)  # 使用最新可用的变换
                pt.point.x, pt.point.y, pt.point.z = offset
                try:
                    pt_torso = self.tf_listener.transformPoint('torso_lift_link', pt)
                    positions.append([pt_torso.point.x, pt_torso.point.y, pt_torso.point.z])
                    radii.append(sphere_radii[len(radii)])
                except Exception as e:
                    rospy.logwarn(f'TF变换失败: {e}')
        return np.array(positions), np.array(radii)

    def find_closest_obstacle(self):
        """返回最近的机械臂球心、障碍物球心、障碍物半径、最小安全距离"""
        # ✅ 使用 TF 变换获取准确的球体位置（与 safety_filter_right.py 一致）
        if len(self.original_obstacles) == 0:
            return None, None, None, 999.0

        robot_pos, robot_r = self.get_collision_spheres_torso_lift_link()
        
        if len(robot_pos) == 0:
            rospy.logwarn("⚠️ 无法获取机械臂碰撞球位置")
            return None, None, None, 999.0

        obs_pos = np.array([o[:3] for o in self.original_obstacles])
        obs_r   = np.array([o[3]  for o in self.original_obstacles])

        deltas = robot_pos[:, None, :] - obs_pos[None, :, :]
        dists  = np.linalg.norm(deltas, axis=2)  # 球心之间的距离
        safe_d = dists - (robot_r[:, None] + obs_r[None, :])  # 安全距离（表面距离）

        i, j = np.unravel_index(np.argmin(safe_d), safe_d.shape)
        
        # # 🔍 详细调试输出
        # rospy.loginfo_throttle(2.0, 
        #     f"🔍 最近碰撞对: 机械臂球{i} @ {robot_pos[i]} (r={robot_r[i]:.3f}m) "
        #     f"<-> 障碍物{j} @ {obs_pos[j]} (r={obs_r[j]:.3f}m)")
        # rospy.loginfo_throttle(2.0,
        #     f"   球心距离 = {dists[i,j]:.3f}m ({dists[i,j]*100:.1f}cm)")
        # rospy.loginfo_throttle(2.0,
        #     f"   半径总和 = {robot_r[i]:.3f}m + {obs_r[j]:.3f}m = {robot_r[i]+obs_r[j]:.3f}m ({(robot_r[i]+obs_r[j])*100:.1f}cm)")
        # rospy.loginfo_throttle(2.0,
        #     f"   表面距离 = 球心距 - 半径和 = {dists[i,j]:.3f} - {robot_r[i]+obs_r[j]:.3f} = {safe_d[i,j]:.3f}m ({safe_d[i,j]*100:.1f}cm)")
        
        # # ⚠️ 如果真的碰撞了但距离还是正数，说明坐标系或位置有问题
        # if safe_d[i,j] > 0.10:  # 表面距离 > 10cm
        #     rospy.logwarn_throttle(2.0, 
        #         f"⚠️ 警告: 表面距离 {safe_d[i,j]*100:.1f}cm 看起来很大！"
        #         f"可能原因: 1) 机械臂碰撞球位置定义错误 2) 障碍物位置坐标系不一致")
        
        return robot_pos[i], obs_pos[j], obs_r[j], safe_d[i, j]

    def compute_ik_step(self, target_cartesian_pos):
        """使用 MoveIt IK 计算朝向某个笛卡尔目标的小步关节速度（最可靠）"""
        ee = self.get_end_effector_position()
        if ee is None:
            return np.zeros(7)

        direction = target_cartesian_pos - ee
        dist = np.linalg.norm(direction)
        if dist < 0.005:
            return np.zeros(7)

        direction = direction / dist
        step = min(0.02, dist)  # 每步最多 2cm（降低单步距离，提高安全性）
        intermediate = ee + direction * step

        quat = self.get_current_quat()

        req = GetPositionIKRequest()
        req.ik_request.group_name = "arm_right"
        req.ik_request.robot_state.joint_state.name = self.arm_right_joint_names
        req.ik_request.robot_state.joint_state.position = self.current_q_right.tolist()
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = rospy.Duration(0.05)
        req.ik_request.pose_stamped.header.frame_id = "torso_lift_link"
        req.ik_request.pose_stamped.header.stamp = rospy.Time.now()
        req.ik_request.pose_stamped.pose.position = Point(*intermediate)
        req.ik_request.pose_stamped.pose.orientation.x = quat[0]
        req.ik_request.pose_stamped.pose.orientation.y = quat[1]
        req.ik_request.pose_stamped.pose.orientation.z = quat[2]
        req.ik_request.pose_stamped.pose.orientation.w = quat[3]

        try:
            resp = self.ik_service(req)
            if resp.error_code.val == 1:
                q_goal = np.array(resp.solution.joint_state.position[:7])
                qdot = (q_goal - self.current_q_right) / self.dt
                return np.clip(qdot, -0.8, 0.8)
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"IK failed: {e}")
        return np.zeros(7)

    def pos_callback_right(self, msg):
        self.target_q_right = np.array(msg.data)
        self.motion_blocked = False
        self.command_sent = False
        self.escape_retry_count = 0
        rospy.loginfo(f"收到新目标关节角: {self.target_q_right}")

    def joint_callback(self, msg):
        try:
            idx = [msg.name.index(j) for j in self.arm_right_joint_names]
            self.current_q_right = np.array([msg.position[i] for i in idx])
        except:
            pass

    def sphere_callback(self, msg):
        self.original_obstacles = [msg.data[i:i+4] for i in range(0, len(msg.data), 4)]
        
        # 调试输出：检查接收到的障碍物数据
        if len(self.original_obstacles) > 0:
            rospy.loginfo_throttle(3.0, 
                f"📦 接收到 {len(self.original_obstacles)} 个障碍物 (torso_lift_link坐标系):")
            for idx, obs in enumerate(self.original_obstacles):
                rospy.loginfo_throttle(3.0, 
                    f"   障碍物{idx}: pos=[{obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}], r={obs[3]:.3f}m")
        
        obs_arr = np.array(self.original_obstacles) if self.original_obstacles else np.empty((0,4))
        self.cbf_right = CBF.from_config(CollisionsVelocityConfig(
            robot=self.right_arm,
            collision_positions=obs_arr[:, :3] if obs_arr.size else np.array([]),
            collision_radii=obs_arr[:, 3] if obs_arr.size else np.array([])
        ))

    def update(self, event):
        if rospy.Time.now().to_sec() % 2 < 0.2:  # 每2秒更新一次目标点
            self.update_target_goal_transform()

        if not self.reached_initial_position:
            if np.all(np.abs(self.current_q_right - self.target_q_right) < self.position_tolerance):
                self.reached_initial_position = True
                rospy.loginfo("已到达初始位姿，安全滤波器正式启用")
            return

        # ==================== 输出最近障碍物距离 ====================
        _, _, _, min_safe_dist = self.find_closest_obstacle()
        if min_safe_dist < 999.0:  # 有障碍物存在
            rospy.loginfo_throttle(1.0, f"⚠️  最近障碍物距离: {min_safe_dist*100:.1f}cm")
        else:
            rospy.loginfo_throttle(5.0, "✓ 无障碍物检测到")

        # ==================== 被卡死时的逃逸逻辑 ====================
        if self.motion_blocked:
            rospy.logwarn_throttle(1.0, "机械臂被卡死 → 执行逃逸动作（仅远离障碍物5cm）")

            robot_pt, obs_pt, _, safe_dist = self.find_closest_obstacle()

            if safe_dist > 0.08:  # 已经比较安全了（8cm），直接朝目标走
                u_nom = self.compute_ik_step(self.target_goal)
            else:
                # 计算远离方向（从障碍物指向机械臂）
                repel_dir = (robot_pt - obs_pt)
                repel_dir /= (np.linalg.norm(repel_dir) + 1e-8)
                escape_target = self.get_end_effector_position() + repel_dir * 0.05  # 仅远离 5cm

                u_nom = self.compute_ik_step(escape_target)
                rospy.loginfo_throttle(1.0, f"🔍 逃逸 IK: u_nom 模长={np.linalg.norm(u_nom):.4f}")

            # ⚠️ 逃逸时禁用 CBF，强制执行远离动作
            if safe_dist <= 0.08:  # 距离太近，禁用 CBF
                u_safe = u_nom  # 直接使用 IK 结果，不经过 CBF
                rospy.logwarn_throttle(1.0, f"🚨 逃逸模式：禁用 CBF，强制远离 (距离={safe_dist*100:.1f}cm)")
            else:  # 距离已经比较安全，可以使用 CBF
                u_safe = self.cbf_right.safety_filter(self.current_q_right, u_nom)
                rospy.loginfo_throttle(1.0, "逃逸中，使用 CBF 保护")
                
            u_final = self.limit_acceleration(u_safe)

            # 执行逃逸动作
            new_q = self.current_q_right + u_final * self.dt
            self.publish_trajectory(new_q, u_final)
            self.escape_retry_count += 1
            
            rospy.loginfo_throttle(1.0, 
                f"逃逸中... 第 {self.escape_retry_count} 次 | "
                f"距离={safe_dist*100:.1f}cm | "
                f"|u_nom|={np.linalg.norm(u_nom):.3f} → |u_safe|={np.linalg.norm(u_safe):.3f} → |u_final|={np.linalg.norm(u_final):.3f}")
            
            # ✅ 根据距离判断是否恢复正常模式
            if safe_dist >= 0.10:  # 距离大于等于 10cm
                rospy.loginfo("✓ 距离已安全 (≥10cm)，恢复正常跟踪模式")
                self.motion_blocked = False
                self.command_sent = False
                self.escape_retry_count = 0
            
            return

        # ==================== 正常跟踪模式 ====================
        if self.command_sent:
            return
        self.command_sent = True

        # ✅ 计算与障碍物的距离，判断是否需要 CBF（使用 TF 变换获取准确位置）
        robot_pos, robot_r = self.get_collision_spheres_torso_lift_link()
        
        cbf_active = False
        min_h = 999.0
        
        if len(self.original_obstacles) > 0 and len(robot_pos) > 0:
            obs_pos = np.array([o[:3] for o in self.original_obstacles])
            obs_r = np.array([o[3] for o in self.original_obstacles])
            
            # 计算所有机械臂球与障碍物球之间的距离（与 safety_filter_right.py 逻辑一致）
            deltas = robot_pos[:, None, :] - obs_pos[None, :, :]
            dists = np.linalg.norm(deltas, axis=2)
            h_values = dists - (robot_r[:, None] + obs_r[None, :])
            
            min_h = np.min(h_values)
            
            # CBF 激活阈值（提高激活距离，提前介入）
            if min_h <= 0.30:  # 30cm 以内激活 CBF
                cbf_active = True
                rospy.loginfo_throttle(1.0, f"🛡️ CBF 激活 (最近距离={min_h*100:.1f}cm)")
            
            # 检测即将碰撞（提高阈值，更早触发逃逸）
            if min_h < 0.05:  # 5cm 以内认为碰撞即将发生（从 2cm 增加到 5cm）
                rospy.logerr(f"⚠️ 碰撞即将发生 (距离={min_h*100:.1f}cm) → 进入逃逸模式")
                self.motion_blocked = True
                self.escape_retry_count = 0
                return

        # 生成名义控制指令
        u_nom = (self.target_q_right - self.current_q_right) / self.dt
        
        # 根据是否有障碍物接近，决定是否使用 CBF
        if cbf_active:
            u_safe = self.cbf_right.safety_filter(self.current_q_right, u_nom)
            rospy.loginfo_throttle(1.0, f"CBF 过滤: |u_nom|={np.linalg.norm(u_nom):.3f} → |u_safe|={np.linalg.norm(u_safe):.3f}")
        else:
            u_safe = u_nom
            rospy.loginfo_throttle(2.0, "✓ 无障碍物接近，直接执行命令")
        
        u_final = self.limit_acceleration(u_safe)

        # 判断是否被完全卡死（只在 CBF 激活时检查）
        # if cbf_active and np.linalg.norm(u_final) < 0.02 and np.linalg.norm(u_nom) > 0.1:
        #     rospy.logerr("CBF完全阻塞运动 → 进入逃逸模式")
        #     self.motion_blocked = True
        #     self.escape_retry_count = 0
        #     return

        new_q = self.current_q_right + u_final * self.dt
        self.publish_trajectory(new_q, u_final)

    def publish_trajectory(self, positions, velocities=np.zeros(7)):
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = self.arm_right_joint_names
        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        point.velocities = velocities.tolist()
        point.time_from_start = rospy.Duration(self.dt)
        traj.points = [point]
        self.arm_pub_right.publish(traj)

def main():
    SafetyFilterWithGoal()
    rospy.spin()

if __name__ == '__main__':
    main()