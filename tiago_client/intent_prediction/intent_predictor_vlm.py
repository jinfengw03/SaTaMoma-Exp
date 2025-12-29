#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time Intent Predictor using Camera and Action Commands
基于摄像头和动作指令的实时意图预测器
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, String
from cv_bridge import CvBridge
import subprocess
import tempfile
import os
from collections import deque
import threading
import tf
from geometry_msgs.msg import PointStamped

class IntentPredictorVLM:
    def __init__(self):
        rospy.init_node('intent_predictor_vlm_node')
        
        # 参数
        self.model_name = rospy.get_param('~model_name', 'llava:7b')
        self.analysis_interval = rospy.get_param('~analysis_interval', 2.0)  # 每2秒分析一次
        self.frame_buffer_size = rospy.get_param('~frame_buffer_size', 30)  # 保留最近30帧
        self.enable_ollama = rospy.get_param('~enable_ollama', True)  # 是否启用 Ollama
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 状态变量
        self.latest_image = None
        self.frame_buffer = deque(maxlen=self.frame_buffer_size)
        self.current_joint_positions = None
        self.current_arm_command = None
        self.last_analysis_time = rospy.Time.now()
        self.prediction_lock = threading.Lock()
        
        # 语音输入
        self.latest_speech = None
        self.speech_timestamp = None
        self.speech_timeout = rospy.Duration(30.0)  # 语音有效期30秒
        
        # 球体检测信息
        self.detected_spheres = None
        self.sphere_timestamp = None
        
        # 末端执行器位置跟踪（用于距离变化检测）
        self.ee_sphere_positions = None  # 末端执行器碰撞球体位置
        self.ee_sphere_radii = None      # 末端执行器碰撞球体半径
        self.previous_min_distance = None  # 上一帧的最小距离
        self.current_min_distance = None   # 当前帧的最小距离
        self.distance_history = deque(maxlen=5)  # 保留最近5帧的距离变化
        
        # 动作历史
        self.action_history = deque(maxlen=10)
        
        # 订阅器
        self.image_sub = rospy.Subscriber(
            '/xtion/rgb/image_raw', 
            Image, 
            self.image_callback, 
            queue_size=1
        )
        
        self.joint_sub = rospy.Subscriber(
            '/joint_states', 
            JointState, 
            self.joint_callback, 
            queue_size=1
        )
        
        self.arm_command_sub = rospy.Subscriber(
            '/custom_arm_pos_right', 
            Float64MultiArray, 
            self.arm_command_callback, 
            queue_size=1
        )
        
        # 语音输入订阅器
        self.speech_sub = rospy.Subscriber(
            '/speech_input',
            String,
            self.speech_callback,
            queue_size=1
        )
        
        # 球体检测订阅器
        self.sphere_sub = rospy.Subscriber(
            '/detected_spheres',
            Float64MultiArray,
            self.sphere_callback,
            queue_size=1
        )
        
        # 发布器
        self.intent_pub = rospy.Publisher('/predicted_intent', String, queue_size=10)
        self.debug_image_pub = rospy.Publisher('/intent_debug_image', Image, queue_size=1)
        
        # 临时文件目录
        self.temp_dir = tempfile.mkdtemp()
        rospy.loginfo(f'Temporary directory: {self.temp_dir}')
        
        # TF Listener（用于获取末端执行器球体位置）
        self.tf_listener = None
        try:
            import tf
            self.tf_listener = tf.TransformListener()
            rospy.loginfo('TF Listener initialized for end-effector tracking')
        except Exception as e:
            rospy.logwarn(f'TF Listener initialization failed: {e}')
        
        # 预定义末端执行器球体（arm_right_6_link 和 arm_right_7_link）
        # 这些是最接近夹爪的球体
        self.ee_link_names = ['arm_right_6_link', 'arm_right_7_link']
        self.ee_sphere_offsets = [
            [(0.09, 0.0, 0.0), (0.15, 0.0, 0.0)],  # arm_right_6_link: 2个球体
            [(0.0, 0.0, 0.0)]                       # arm_right_7_link: 1个球体
        ]
        self.ee_sphere_radii_def = [0.07, 0.07, 0.07]  # 对应3个球体的半径
        
        # 检查 Ollama（必需）
        if not self.check_ollama():
            rospy.logfatal('Ollama not available. Please install Ollama and run: ollama pull llava:7b')
            rospy.signal_shutdown('Ollama required but not available')
            return
        
        rospy.loginfo('VLM Intent Predictor Node initialized')
        
        # 启动分析线程
        self.analysis_thread = threading.Thread(target=self.analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
    def check_ollama(self):
        """检查 Ollama 是否可用"""
        try:
            result = subprocess.run(
                ["ollama", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            rospy.loginfo(f'Ollama version: {result.stdout.strip()}')
            
            # 检查模型
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if self.model_name not in result.stdout:
                rospy.logwarn(f'Model {self.model_name} not found. Run: ollama pull {self.model_name}')
                return False
            return True
        except Exception as e:
            rospy.logwarn(f'Ollama check failed: {e}')
            return False
    
    def image_callback(self, msg):
        """接收摄像头图像"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image
            
            # 添加到帧缓冲区
            timestamp = rospy.Time.now()
            self.frame_buffer.append((timestamp, cv_image))
            
        except Exception as e:
            rospy.logwarn_throttle(5.0, f'Image conversion error: {e}')
    
    def joint_callback(self, msg):
        """接收关节状态"""
        try:
            arm_right_joints = [
                'arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint',
                'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint',
                'arm_right_7_joint'
            ]
            idx = [msg.name.index(j) for j in arm_right_joints]
            self.current_joint_positions = np.array([msg.position[i] for i in idx])
        except (ValueError, IndexError) as e:
            rospy.logwarn_throttle(5.0, f'Joint extraction error: {e}')
    
    def arm_command_callback(self, msg):
        """接收机械臂命令"""
        self.current_arm_command = np.array(msg.data)
        self.action_history.append(self.current_arm_command.copy())
    
    def speech_callback(self, msg):
        """接收语音输入"""
        self.latest_speech = msg.data
        self.speech_timestamp = rospy.Time.now()
        rospy.loginfo(f'Received speech input: "{self.latest_speech}"')
    
    def sphere_callback(self, msg):
        """接收球体检测信息"""
        # 数据格式: [x1, y1, z1, r1, x2, y2, z2, r2, ...]
        if len(msg.data) % 4 != 0:
            rospy.logwarn(f'Invalid sphere data length: {len(msg.data)}')
            return
        
        num_spheres = len(msg.data) // 4
        spheres = []
        for i in range(num_spheres):
            idx = i * 4
            spheres.append({
                'x': msg.data[idx],
                'y': msg.data[idx + 1],
                'z': msg.data[idx + 2],
                'r': msg.data[idx + 3]
            })
        
        self.detected_spheres = spheres
        self.sphere_timestamp = rospy.Time.now()
        rospy.loginfo(f'Received {num_spheres} detected spheres')
        
        # 更新末端执行器与目标物体的距离
        self.update_ee_target_distance()
    
    def get_ee_sphere_positions(self):
        """获取末端执行器碰撞球体在 torso_lift_link 坐标系下的位置"""
        if self.tf_listener is None:
            return None, None
        
        positions = []
        radii = []
        target_frame = 'torso_lift_link'
        
        try:
            import tf
            from geometry_msgs.msg import PointStamped
            
            radius_idx = 0
            for link_name, offsets in zip(self.ee_link_names, self.ee_sphere_offsets):
                for offset in offsets:
                    pt = PointStamped()
                    pt.header.frame_id = link_name
                    pt.header.stamp = rospy.Time(0)
                    pt.point.x, pt.point.y, pt.point.z = offset
                    
                    try:
                        self.tf_listener.waitForTransform(
                            target_frame, link_name, pt.header.stamp, rospy.Duration(0.5)
                        )
                        pt_transformed = self.tf_listener.transformPoint(target_frame, pt)
                        pos = [pt_transformed.point.x, pt_transformed.point.y, pt_transformed.point.z]
                        positions.append(pos)
                        radii.append(self.ee_sphere_radii_def[radius_idx])
                        radius_idx += 1
                    except Exception as e:
                        rospy.logdebug(f'TF transform failed for {link_name}: {e}')
                        radius_idx += 1
                        continue
            
            if len(positions) > 0:
                return np.array(positions), np.array(radii)
            else:
                return None, None
                
        except Exception as e:
            rospy.logwarn_throttle(5.0, f'Failed to get EE sphere positions: {e}')
            return None, None
    
    def calculate_min_distance_to_targets(self, ee_positions, ee_radii, target_spheres):
        """计算末端执行器球体到目标球体的最小距离（表面到表面）"""
        if ee_positions is None or target_spheres is None or len(target_spheres) == 0:
            return None
        
        min_distance = float('inf')
        closest_pair = None
        
        for ee_pos, ee_r in zip(ee_positions, ee_radii):
            for target in target_spheres:
                target_pos = np.array([target['x'], target['y'], target['z']])
                target_r = target['r']
                
                # 中心距离
                center_dist = np.linalg.norm(ee_pos - target_pos)
                # 表面到表面的距离
                surface_dist = center_dist - ee_r - target_r
                
                if surface_dist < min_distance:
                    min_distance = surface_dist
                    closest_pair = {
                        'ee_pos': ee_pos.tolist(),
                        'target_pos': target_pos.tolist(),
                        'distance': surface_dist
                    }
        
        return min_distance, closest_pair
    
    def update_ee_target_distance(self):
        """更新末端执行器到目标物体的距离，并记录变化趋势"""
        # 获取末端执行器球体位置
        ee_positions, ee_radii = self.get_ee_sphere_positions()
        if ee_positions is None:
            rospy.logdebug('Cannot get EE positions')
            return
        
        self.ee_sphere_positions = ee_positions
        self.ee_sphere_radii = ee_radii
        
        # 计算到目标的最小距离
        if self.detected_spheres is not None and len(self.detected_spheres) > 0:
            min_dist, closest_pair = self.calculate_min_distance_to_targets(
                ee_positions, ee_radii, self.detected_spheres
            )
            
            if min_dist is not None:
                # 更新距离历史
                self.previous_min_distance = self.current_min_distance
                self.current_min_distance = min_dist
                
                # 计算距离变化
                if self.previous_min_distance is not None:
                    distance_change = self.current_min_distance - self.previous_min_distance
                    self.distance_history.append(distance_change)
                    
                    if distance_change < 0:
                        rospy.loginfo(f'末端执行器正在接近目标: {abs(distance_change)*100:.1f}cm closer (当前距离: {min_dist*100:.1f}cm)')
                    elif distance_change > 0:
                        rospy.loginfo(f'末端执行器正在远离目标: {distance_change*100:.1f}cm farther (当前距离: {min_dist*100:.1f}cm)')
                else:
                    rospy.loginfo(f'初始末端执行器到目标距离: {min_dist*100:.1f}cm')
    
    def get_distance_trend_score(self):
        """
        计算距离变化趋势评分 (0-1)
        接近目标 = 高分，远离目标 = 低分
        """
        if len(self.distance_history) < 2:
            return 0.5  # 中性分数（数据不足）
        
        # 计算平均距离变化
        avg_change = np.mean(list(self.distance_history))
        
        # 标准化：-0.1m (接近很快) -> 1.0分，+0.1m (远离很快) -> 0.0分
        # 使用 sigmoid 函数映射
        score = 1.0 / (1.0 + np.exp(avg_change * 20))  # 系数20控制灵敏度
        
        return score
    
    def analyze_sphere_patterns(self, spheres):
        """分析球体的空间分布模式，推测可能的物体类型"""
        if not spheres or len(spheres) == 0:
            return []
        
        patterns = []
        n = len(spheres)
        
        # 按距离分组（找到最近的球体）
        nearest_spheres = sorted(spheres, key=lambda s: np.sqrt(s['x']**2 + s['y']**2 + s['z']**2))[:5]
        
        # 模式1: 垂直堆叠 (例如：竖着的薯片罐、瓶子)
        # 条件：x, y 坐标相近，z 坐标递增
        for i in range(len(nearest_spheres)):
            vertical_group = [nearest_spheres[i]]
            base_sphere = nearest_spheres[i]
            
            for j in range(len(nearest_spheres)):
                if i == j:
                    continue
                candidate = nearest_spheres[j]
                
                # 检查 x, y 是否接近（误差 < 0.1m）
                dx = abs(candidate['x'] - base_sphere['x'])
                dy = abs(candidate['y'] - base_sphere['y'])
                dz = candidate['z'] - base_sphere['z']
                
                if dx < 0.1 and dy < 0.1 and 0 < dz < 0.3:  # z 方向上方 30cm 内
                    vertical_group.append(candidate)
            
            if len(vertical_group) >= 2:
                # 计算组的高度和位置
                z_coords = [s['z'] for s in vertical_group]
                height = max(z_coords) - min(z_coords)
                avg_x = np.mean([s['x'] for s in vertical_group])
                avg_y = np.mean([s['y'] for s in vertical_group])
                avg_z = np.mean(z_coords)
                avg_r = np.mean([s['r'] for s in vertical_group])
                
                patterns.append({
                    'type': 'vertical_stack',
                    'description': f'{len(vertical_group)} spheres stacked vertically',
                    'possible_objects': ['bottle', 'can', 'cylinder container', 'vertical snack tube (like Pringles)'],
                    'height': height,
                    'position': (avg_x, avg_y, avg_z),
                    'avg_radius': avg_r,
                    'sphere_count': len(vertical_group)
                })
                break  # 只报告一个垂直堆叠模式
        
        # 模式2: 水平排列 (例如：横放的物体)
        # 条件：z 坐标相近，x 或 y 坐标递增
        for i in range(len(nearest_spheres)):
            horizontal_group = [nearest_spheres[i]]
            base_sphere = nearest_spheres[i]
            
            for j in range(len(nearest_spheres)):
                if i == j:
                    continue
                candidate = nearest_spheres[j]
                
                dz = abs(candidate['z'] - base_sphere['z'])
                dx = abs(candidate['x'] - base_sphere['x'])
                dy = abs(candidate['y'] - base_sphere['y'])
                
                # z 相近（误差 < 0.05m），x 或 y 有较大差异
                if dz < 0.05 and (dx > 0.05 or dy > 0.05) and (dx < 0.3 or dy < 0.3):
                    horizontal_group.append(candidate)
            
            if len(horizontal_group) >= 2:
                avg_x = np.mean([s['x'] for s in horizontal_group])
                avg_y = np.mean([s['y'] for s in horizontal_group])
                avg_z = np.mean([s['z'] for s in horizontal_group])
                
                patterns.append({
                    'type': 'horizontal_line',
                    'description': f'{len(horizontal_group)} spheres arranged horizontally',
                    'possible_objects': ['horizontal bottle', 'elongated object', 'box on its side'],
                    'position': (avg_x, avg_y, avg_z),
                    'sphere_count': len(horizontal_group)
                })
                break
        
        # 模式3: 单个球体（可能是球形物体）
        if len(patterns) == 0 and len(nearest_spheres) > 0:
            for sphere in nearest_spheres[:3]:  # 最多报告3个最近的单个球体
                patterns.append({
                    'type': 'single_sphere',
                    'description': 'Single isolated sphere',
                    'possible_objects': ['ball', 'apple', 'orange', 'spherical object'],
                    'position': (sphere['x'], sphere['y'], sphere['z']),
                    'radius': sphere['r']
                })
        
        # 模式4: 聚集模式（多个球体聚集，可能是复杂物体）
        if len(nearest_spheres) >= 4:
            # 计算所有球体的中心
            center_x = np.mean([s['x'] for s in nearest_spheres])
            center_y = np.mean([s['y'] for s in nearest_spheres])
            center_z = np.mean([s['z'] for s in nearest_spheres])
            
            # 检查是否聚集（所有球体到中心距离 < 0.2m）
            distances = [np.sqrt((s['x']-center_x)**2 + (s['y']-center_y)**2 + (s['z']-center_z)**2) 
                        for s in nearest_spheres]
            if max(distances) < 0.2:
                patterns.append({
                    'type': 'cluster',
                    'description': f'{len(nearest_spheres)} spheres clustered together',
                    'possible_objects': ['complex object', 'multiple small items', 'irregular shape'],
                    'position': (center_x, center_y, center_z),
                    'sphere_count': len(nearest_spheres)
                })
        
        return patterns
    
    def save_frame_to_file(self, image, filename):
        """保存帧到临时文件"""
        filepath = os.path.join(self.temp_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath
    
    def query_llava(self, image_path, prompt):
        """使用 Ollama 查询 LLaVA"""
        try:
            full_prompt = f"{prompt}\n{image_path}"
            result = subprocess.run(
                ["ollama", "run", self.model_name, full_prompt],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30  # 30秒超时
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return f"Error: {error_msg}"
        
        except subprocess.TimeoutExpired:
            return "Error: LLaVA query timeout (30s)"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_context_prompt(self):
        """生成包含动作上下文的提示"""
        prompt_parts = [
            "You are analyzing a robot teleoperation scenario.",
            "Describe what you see in the image and predict the operator's intent.",
            ""
        ]
        
        # 添加球体检测和模式分析
        if self.detected_spheres is not None and len(self.detected_spheres) > 0:
            patterns = self.analyze_sphere_patterns(self.detected_spheres)
            
            if patterns:
                prompt_parts.append("=== Detected 3D Object Patterns from Depth Analysis ===")
                prompt_parts.append(f"Total spheres detected: {len(self.detected_spheres)}")
                prompt_parts.append("")
                
                for idx, pattern in enumerate(patterns, 1):
                    prompt_parts.append(f"Pattern {idx}: {pattern['type'].upper()}")
                    prompt_parts.append(f"  Description: {pattern['description']}")
                    
                    pos = pattern['position']
                    prompt_parts.append(f"  Position: ({pos[0]:.2f}m, {pos[1]:.2f}m, {pos[2]:.2f}m)")
                    
                    # 计算距离
                    distance = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                    prompt_parts.append(f"  Distance from robot: {distance:.2f}m")
                    
                    if 'height' in pattern:
                        prompt_parts.append(f"  Height: {pattern['height']:.2f}m")
                    if 'avg_radius' in pattern:
                        prompt_parts.append(f"  Average radius: {pattern['avg_radius']:.3f}m")
                    
                    prompt_parts.append(f"  Likely objects: {', '.join(pattern['possible_objects'])}")
                    prompt_parts.append("")
                
                prompt_parts.append("IMPORTANT: Use this 3D structure information to identify objects in the image.")
                prompt_parts.append("For example:")
                prompt_parts.append("  - Vertical stack (3+ spheres) → Bottle, Can, Pringles tube")
                prompt_parts.append("  - Single sphere → Ball, Apple, Orange")
                prompt_parts.append("  - Horizontal line → Lying bottle, Box")
                prompt_parts.append("")
        
        # 添加末端执行器距离变化信息
        if self.current_min_distance is not None:
            prompt_parts.append("=== End-Effector to Target Distance ===")
            prompt_parts.append(f"Current minimum distance: {self.current_min_distance*100:.1f}cm")
            
            if self.previous_min_distance is not None:
                distance_change = self.current_min_distance - self.previous_min_distance
                if distance_change < -0.01:  # 接近 >1cm
                    prompt_parts.append(f"Trend: APPROACHING target ({abs(distance_change)*100:.1f}cm closer)")
                    prompt_parts.append("This suggests the operator is trying to reach/grasp the object.")
                elif distance_change > 0.01:  # 远离 >1cm
                    prompt_parts.append(f"Trend: MOVING AWAY from target ({distance_change*100:.1f}cm farther)")
                    prompt_parts.append("This suggests the operator may be repositioning or aborting.")
                else:
                    prompt_parts.append("Trend: STABLE (no significant movement)")
            
            # 显示距离变化历史趋势
            if len(self.distance_history) >= 2:
                avg_change = np.mean(list(self.distance_history))
                if avg_change < -0.005:
                    prompt_parts.append("Overall trend: Consistently approaching (strong grasping intent)")
                elif avg_change > 0.005:
                    prompt_parts.append("Overall trend: Consistently moving away (exploring/repositioning)")
                else:
                    prompt_parts.append("Overall trend: Hovering near target (preparing to grasp)")
            prompt_parts.append("")
        
        # 添加语音输入（如果有效）
        if self.latest_speech is not None and self.speech_timestamp is not None:
            time_since_speech = rospy.Time.now() - self.speech_timestamp
            if time_since_speech < self.speech_timeout:
                prompt_parts.append(f'Human speech input: "{self.latest_speech}"')
                prompt_parts.append(f'(spoken {time_since_speech.to_sec():.1f} seconds ago)')
                prompt_parts.append('IMPORTANT: Consider this speech input when predicting intent.')
                prompt_parts.append('For example: "I am hungry" suggests intent to fetch food.')
                prompt_parts.append('')
        
        # 添加关节信息
        if self.current_joint_positions is not None:
            prompt_parts.append(f"Current joint positions: {self.current_joint_positions.tolist()}")
        
        # 添加动作历史
        if len(self.action_history) > 0:
            recent_velocity = np.linalg.norm(
                self.action_history[-1] - self.action_history[0]
            ) if len(self.action_history) > 1 else 0.0
            
            if recent_velocity < 0.01:
                motion_state = "stationary"
            elif recent_velocity < 0.1:
                motion_state = "slow movement"
            else:
                motion_state = "fast movement"
            
            prompt_parts.append(f"Arm motion state: {motion_state}")
        
        prompt_parts.extend([
            "",
            "Please provide:",
            "1. Identify objects in the scene, USING the 3D pattern information to help recognition",
            "2. Match detected patterns (vertical stack, single sphere, etc.) to visible objects",
            "3. Current robot action based on arm motion",
            "4. Predicted operator intent considering:",
            "   - Speech input (if any)",
            "   - Which object pattern is closest",
            "   - Object type and arm motion",
            "   - End-effector distance trend (approaching = grasping intent)",
            "5. Confidence level (High/Medium/Low)",
            "6. If confidence is high, suggest next steps to complete the task",
        ])
        
        return "\n".join(prompt_parts)
    
    def calculate_weighted_confidence(self, vlm_response):
        """
        计算加权置信度
        - 基于上下文和语音输入: 70%
        - 基于物体与机器人距离: 20%
        - 末端执行器相对位置变化: 10%
        """
        # 1. 上下文和语音输入置信度 (70%)
        context_score = 0.5  # 默认中性
        
        # 检查VLM响应中的置信度关键词
        response_lower = vlm_response.lower()
        if 'high confidence' in response_lower or 'very confident' in response_lower:
            context_score = 0.9
        elif 'medium confidence' in response_lower or 'moderate' in response_lower:
            context_score = 0.6
        elif 'low confidence' in response_lower or 'uncertain' in response_lower:
            context_score = 0.3
        
        # 如果有语音输入，提高上下文置信度
        if self.latest_speech is not None and self.speech_timestamp is not None:
            time_since_speech = rospy.Time.now() - self.speech_timestamp
            if time_since_speech < self.speech_timeout:
                context_score = min(1.0, context_score + 0.2)  # 提升20%

        # 2. 物体与机器人距离置信度 (20%)
        distance_score = 0.5  # 默认中性
        
        if self.current_min_distance is not None:
            # 距离越近，置信度越高（假设目标是接近物体）
            # 使用指数衰减：0.1m -> 0.9分，0.5m -> 0.5分，1.0m -> 0.2分
            distance_score = np.exp(-self.current_min_distance * 2.0)
            distance_score = max(0.1, min(1.0, distance_score))

        # 3. 末端执行器相对位置变化置信度 (10%)
        ee_movement_score = self.get_distance_trend_score()
        
        # 加权计算总置信度
        weights = {
            'context': 0.70,
            'distance': 0.20,
            'ee_movement': 0.10
        }
        
        total_confidence = (
            context_score * weights['context'] +
            distance_score * weights['distance'] +
            ee_movement_score * weights['ee_movement']
        )
        
        # 生成置信度报告
        confidence_report = {
            'total': total_confidence,
            'breakdown': {
                'context_and_speech': context_score,
                'object_distance': distance_score,
                'ee_movement_trend': ee_movement_score
            },
            'weights': weights
        }
        
        return confidence_report
    
    def format_prediction_with_confidence(self, vlm_response, confidence_report):
        """格式化预测结果，包含置信度信息"""
        total_conf = confidence_report['total']
        breakdown = confidence_report['breakdown']
        weights = confidence_report['weights']
        
        # 确定置信度等级
        if total_conf >= 0.70:
            confidence_level = "HIGH"
        elif total_conf >= 0.50:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        report_lines = [
            "="*70,
            "INTENT PREDICTION REPORT",
            "="*70,
            "",
            "VLM Analysis:",
            "-" * 70,
            vlm_response,
            "",
            "-" * 70,
            "WEIGHTED CONFIDENCE ANALYSIS:",
            "-" * 70,
            f"Overall Confidence: {confidence_level} ({total_conf*100:.1f}%)",
            "",
            "Confidence Breakdown:",
            f"  • Context & Speech (70%):     {breakdown['context_and_speech']*100:.1f}% → Weighted: {breakdown['context_and_speech']*weights['context']*100:.1f}%",
            f"  • Object Distance (20%):      {breakdown['object_distance']*100:.1f}% → Weighted: {breakdown['object_distance']*weights['distance']*100:.1f}%",
            f"  • EE Movement Trend (10%):    {breakdown['ee_movement_trend']*100:.1f}% → Weighted: {breakdown['ee_movement_trend']*weights['ee_movement']*100:.1f}%",
            "",
        ]
        
        # 添加详细解释
        report_lines.append("Detailed Metrics:")
        
        # 距离信息
        if self.current_min_distance is not None:
            report_lines.append(f"  • Current EE-to-target distance: {self.current_min_distance*100:.1f}cm")
            if self.previous_min_distance is not None:
                change = self.current_min_distance - self.previous_min_distance
                trend = "APPROACHING" if change < 0 else ("RETREATING" if change > 0 else "STABLE")
                report_lines.append(f"  • Distance change: {change*100:.1f}cm ({trend})")
        
        # 语音信息
        if self.latest_speech is not None and self.speech_timestamp is not None:
            time_since = rospy.Time.now() - self.speech_timestamp
            if time_since < self.speech_timeout:
                report_lines.append(f'  • Active speech input: "{self.latest_speech}" ({time_since.to_sec():.1f}s ago)')
        
        # 置信度建议
        report_lines.append("")
        report_lines.append("Recommendation:")
        if total_conf >= 0.70:
            report_lines.append("  ✓ HIGH confidence - Intent prediction is reliable")
            report_lines.append("  ✓ Safe to use for predictive assistance or safety checks")
        elif total_conf >= 0.50:
            report_lines.append("  ⚠ MEDIUM confidence - Intent prediction is moderately reliable")
            report_lines.append("  ⚠ Consider waiting for more data or user confirmation")
        else:
            report_lines.append("  ✗ LOW confidence - Intent prediction is uncertain")
            report_lines.append("  ✗ Avoid autonomous actions, require explicit user input")
        
        report_lines.append("="*70)
        
        return "\n".join(report_lines)
    
    def analyze_current_state(self):
        """分析当前状态并预测意图"""
        with self.prediction_lock:
            if self.latest_image is None:
                rospy.logwarn_throttle(5.0, 'No image received yet')
                return
            
            rospy.loginfo('Analyzing current state with LLaVA...')
            
            # 使用 VLM 分析
            image_path = self.save_frame_to_file(
                self.latest_image, 
                f'frame_{rospy.Time.now().to_sec()}.jpg'
            )
            
            prompt = self.generate_context_prompt()
            response = self.query_llava(image_path, prompt)
            
            # 清理临时文件
            try:
                os.remove(image_path)
            except:
                pass
            
            # 计算加权置信度
            confidence_report = self.calculate_weighted_confidence(response)
            
            # 生成完整的预测报告
            full_response = self.format_prediction_with_confidence(response, confidence_report)
            
            # 发布预测结果
            intent_msg = String()
            intent_msg.data = full_response
            self.intent_pub.publish(intent_msg)
            
            rospy.loginfo(f'Predicted Intent:\n{full_response}\n{"-"*60}')
            
            # 交互式确认与执行
            self.check_and_prompt_action(response, confidence_report)
            
            # 发布调试图像（带标注）
            if self.latest_image is not None:
                debug_image = self.latest_image.copy()
                
                # 添加文本标注
                intent_lines = response.split('\n')[:3]  # 前3行
                y_offset = 30
                for line in intent_lines:
                    cv2.putText(
                        debug_image, 
                        line[:50],  # 限制长度
                        (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 255, 0), 
                        2
                    )
                    y_offset += 25
                
                try:
                    debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
                    self.debug_image_pub.publish(debug_msg)
                except Exception as e:
                    rospy.logwarn(f'Debug image publish error: {e}')
    
    def analysis_loop(self):
        """定期分析循环"""
        rate = rospy.Rate(1.0 / self.analysis_interval)
        
        while not rospy.is_shutdown():
            try:
                current_time = rospy.Time.now()
                
                # 检查是否到达分析间隔
                if (current_time - self.last_analysis_time).to_sec() >= self.analysis_interval:
                    self.analyze_current_state()
                    self.last_analysis_time = current_time
                
                rate.sleep()
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                rospy.logerr(f'Analysis loop error: {e}')
    
    def cleanup(self):
        """清理临时文件"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            rospy.loginfo('Temporary files cleaned up')
        except:
            pass

    def execute_action(self, action_type):
        """执行预测的动作"""
        if action_type == "APPROACH_TABLE":
            rospy.loginfo("Executing: Approach Table")
            # 使用 subprocess 运行
            try:
                subprocess.run(["rosrun", "tiago_safety", "approach_table.py"], check=True)
            except subprocess.CalledProcessError as e:
                rospy.logerr(f"Failed to run approach_table: {e}")
                
        elif action_type == "PICK_CHIPS":
            rospy.loginfo("Executing: Pick Chips Policy")
            try:
                # 假设工作空间根目录在 ../.. 相对当前脚本位置，或者使用绝对路径
                # 最好使用 rosrun 或者 python 绝对路径
                # 这里假设在 workspace root 运行
                ws_root = os.path.expanduser("~/tiago_dual_public_ws")
                script_path = os.path.join(ws_root, "src/TiagoRobotSkillLearning/run_policy.py")
                subprocess.run(["python3", script_path, "--skill", "pick"], check=True)
            except subprocess.CalledProcessError as e:
                rospy.logerr(f"Failed to run pick policy: {e}")

    def check_and_prompt_action(self, vlm_response, confidence_report):
        """检查意图并请求用户确认"""
        if confidence_report['total'] < 0.6:
            return

        response_lower = vlm_response.lower()
        action_candidate = None
        action_desc = ""

        # 简单的关键词匹配
        if "table" in response_lower and ("approach" in response_lower or "move" in response_lower or "go to" in response_lower):
            action_candidate = "APPROACH_TABLE"
            action_desc = "Move to Table (approach_table.py)"
        elif ("chip" in response_lower or "snack" in response_lower or "food" in response_lower) and ("pick" in response_lower or "grasp" in response_lower or "grab" in response_lower or "eat" in response_lower):
            action_candidate = "PICK_CHIPS"
            action_desc = "Pick Chips (run_policy.py --skill pick)"

        if action_candidate:
            print("\n" + "!"*60)
            print(f">>> PROPOSED ACTION: {action_desc}")
            print(f">>> Confidence: {confidence_report['total']*100:.1f}%")
            print(">>> Do you want to execute this action? (y/n): ", end='', flush=True)
            
            # 简单的阻塞输入 (注意：这会暂停分析循环)
            try:
                # 使用 select 检查是否有输入，避免无限阻塞（可选，但这里我们希望用户确认）
                # 这里直接使用 input()，意味着程序会暂停直到用户输入
                user_input = input().strip().lower()
                if user_input == 'y':
                    self.execute_action(action_candidate)
                else:
                    print(">>> Action cancelled.")
            except Exception as e:
                print(f"Input error: {e}")
            print("!"*60 + "\n")

def main():
    try:
        predictor = IntentPredictorVLM()
        rospy.on_shutdown(predictor.cleanup)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
