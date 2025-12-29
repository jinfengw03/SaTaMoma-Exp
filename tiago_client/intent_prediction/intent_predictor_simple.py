#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified Intent Predictor using only Camera (VLM) and Speech Input
仅基于视觉语言模型(VLM)和语音输入的意图预测器
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from cv_bridge import CvBridge
import subprocess
import tempfile
import os
import threading
from collections import deque

class IntentPredictorVLMSpeech:
    def __init__(self):
        rospy.init_node('intent_predictor_vlm_speech_node')
        
        # 参数
        self.model_name = rospy.get_param('~model_name', 'llava:7b')
        self.analysis_interval = rospy.get_param('~analysis_interval', 3.0)  # 每3秒分析一次
        self.enable_ollama = rospy.get_param('~enable_ollama', True)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 状态变量
        self.latest_image = None
        self.current_joint_positions = None
        self.joint_history = deque(maxlen=30)  # 存储最近30个关节状态用于计算速度
        self.last_analysis_time = rospy.Time.now()
        self.prediction_lock = threading.Lock()
        
        # 动作历史状态
        self.last_executed_action = None
        self.last_action_time = None
        
        # 语音输入
        self.latest_speech = None
        self.speech_timestamp = None
        self.speech_timeout = rospy.Duration(30.0)  # 语音有效期30秒
        
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
        
        # 语音输入订阅器
        self.speech_sub = rospy.Subscriber(
            '/speech_input',
            String,
            self.speech_callback,
            queue_size=1
        )
        
        # 发布器
        self.intent_pub = rospy.Publisher('/predicted_intent', String, queue_size=10)
        self.debug_image_pub = rospy.Publisher('/intent_debug_image', Image, queue_size=1)
        
        # 临时文件目录
        self.temp_dir = tempfile.mkdtemp()
        rospy.loginfo(f'Temporary directory: {self.temp_dir}')
        
        # 检查 Ollama
        if not self.check_ollama():
            rospy.logfatal('Ollama not available. Please install Ollama and run: ollama pull llava:7b')
            rospy.signal_shutdown('Ollama required but not available')
            return
        
        rospy.loginfo('Simple VLM Intent Predictor Node initialized')
        
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
            # 检查是否包含所需关节
            if all(j in msg.name for j in arm_right_joints):
                idx = [msg.name.index(j) for j in arm_right_joints]
                self.current_joint_positions = np.array([msg.position[i] for i in idx])
                
                # 记录历史状态
                self.joint_history.append({
                    'time': rospy.Time.now().to_sec(),
                    'positions': self.current_joint_positions
                })
        except (ValueError, IndexError) as e:
            rospy.logwarn_throttle(5.0, f'Joint extraction error: {e}')

    def speech_callback(self, msg):
        """接收语音输入"""
        self.latest_speech = msg.data
        self.speech_timestamp = rospy.Time.now()
        rospy.loginfo(f'Received speech input: "{self.latest_speech}"')
    
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
                timeout=30
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
        """生成提示词，仅包含视觉和语音信息"""
        prompt_parts = [
            "You are analyzing a robot teleoperation scenario.",
            "Describe what you see in the image and predict the operator's intent."
        ]
        
        # 上下文信息：如果刚刚执行了接近桌子，提示下一步可能是抓取
        if self.last_executed_action == "APPROACH_TABLE" and self.last_action_time:
             if (rospy.Time.now() - self.last_action_time).to_sec() < 60.0:
                 prompt_parts.append("CONTEXT: The robot has just successfully approached the table.")
                 prompt_parts.append("This sequence strongly implies the user now wants to PICK an object (e.g., chips) from the table.")
                 prompt_parts.append("")
        

        
        # 添加语音输入（如果有效）
        if self.latest_speech is not None and self.speech_timestamp is not None:
            time_since_speech = rospy.Time.now() - self.speech_timestamp
            if time_since_speech < self.speech_timeout:
                prompt_parts.append(f'Human speech input: "{self.latest_speech}"')
                prompt_parts.append(f'(spoken {time_since_speech.to_sec():.1f} seconds ago)')
                prompt_parts.append('IMPORTANT: Consider this speech input as the primary indicator of intent.')
                prompt_parts.append('')
        
        # 添加关节运动信息
        motion_desc = "unknown"
        if len(self.joint_history) >= 5:
            start_state = self.joint_history[0]
            end_state = self.joint_history[-1]
            
            # 计算关节角度的最大变化量
            max_diff = np.max(np.abs(end_state['positions'] - start_state['positions']))
            
            # 阈值设为 0.02 弧度 (约 1.15 度)，排除传感器噪声和微小抖动
            if max_diff > 0.02:
                motion_desc = "MOVING"
            else:
                motion_desc = "STATIONARY (not moving)"
        
        if self.current_joint_positions is not None:
            prompt_parts.append(f"Robot Arm Status: {motion_desc}")
            if "MOVING" in motion_desc:
                 prompt_parts.append("The robot arm is currently MOVING. This strongly indicates an active attempt to reach or grasp something.")
            else:
                 prompt_parts.append("The robot arm is currently STATIONARY.")
            prompt_parts.append('')

        prompt_parts.extend([
            "Please provide:",
            "1. Identify key objects in the scene (e.g., table, chips, bottle, etc.)",
            "2. Predicted operator intent based on the visual scene and speech input",
            "3. Confidence level (High/Medium/Low)",
            "4. Suggested action (e.g., Approach Table, Pick Chips, etc.)",
        ])
        
        return "\n".join(prompt_parts)

    def execute_action(self, action_type):
        """执行预测的动作"""
        if action_type == "APPROACH_TABLE":
            rospy.loginfo("Executing: Approach Table")
            try:
                subprocess.run(["rosrun", "tiago_safety", "approach_table.py"], check=True)
                self.last_executed_action = "APPROACH_TABLE"
                self.last_action_time = rospy.Time.now()
                
                # 自动询问是否执行下一步抓取
                print("\n" + "!"*60)
                print(">>> SEQUENCE PROPOSAL: Pick Chips ")
                print(">>> Approach complete. Do you want to execute 'Pick Chips'? (y/n): ", end='', flush=True)
                try:
                    user_input = input().strip().lower()
                    if user_input == 'y':
                        self.execute_action("PICK_CHIPS")
                    else:
                        print(">>> Sequence cancelled.")
                except Exception as e:
                    print(f"Input error: {e}")
                print("!"*60 + "\n")

            except subprocess.CalledProcessError as e:
                rospy.logerr(f"Failed to run approach_table: {e}")
                
        elif action_type == "PICK_CHIPS":
            rospy.loginfo("Executing: Pick Chips")
            try:
                ws_root = os.path.expanduser("~/tiago_dual_public_ws")
                script_path = os.path.join(ws_root, "src/TiagoRobotSkillLearning/replay_motion.py")
                # 默认使用 Motion0.csv，如果需要其他文件可以在这里指定参数
                subprocess.run(["python3", script_path], check=True)
                self.last_executed_action = "PICK_CHIPS"
                self.last_action_time = rospy.Time.now()
            except subprocess.CalledProcessError as e:
                rospy.logerr(f"Failed to run replay_motion: {e}")

    def check_and_prompt_action(self, vlm_response):
        """检查意图并请求用户确认"""
        response_lower = vlm_response.lower()
        action_candidate = None
        action_desc = ""
        confidence_high = False

        # 检查置信度
        if 'high confidence' in response_lower or 'very confident' in response_lower:
            confidence_high = True
        
        # 如果有明确的语音指令，即使VLM置信度不高也提示
        has_speech = (self.latest_speech is not None and 
                     self.speech_timestamp is not None and 
                     (rospy.Time.now() - self.speech_timestamp) < self.speech_timeout)

        if not confidence_high and not has_speech:
            return

        # 简单的关键词匹配
        if "table" in response_lower and ("approach" in response_lower or "move" in response_lower or "go to" in response_lower):
            action_candidate = "APPROACH_TABLE"
            action_desc = "Move to Table (approach_table.py)"
        elif ("chip" in response_lower or "snack" in response_lower or "food" in response_lower) and ("pick" in response_lower or "grasp" in response_lower or "grab" in response_lower or "eat" in response_lower):
            action_candidate = "PICK_CHIPS"
            action_desc = "Pick Chips"

        if action_candidate:
            # 检查是否刚刚执行过该动作 (防止重复提示)
            if self.last_executed_action == action_candidate:
                 if self.last_action_time and (rospy.Time.now() - self.last_action_time).to_sec() < 60.0:
                     rospy.loginfo(f"Skipping prompt for {action_candidate} because it was executed recently.")
                     return

            print("\n" + "!"*60)
            print(f">>> PROPOSED ACTION: {action_desc}")
            if has_speech:
                print(f">>> Based on Speech: \"{self.latest_speech}\"")
            print(">>> Do you want to execute this action? (y/n): ", end='', flush=True)
            
            try:
                # 使用 input() 阻塞等待用户确认
                user_input = input().strip().lower()
                if user_input == 'y':
                    self.execute_action(action_candidate)
                else:
                    print(">>> Action cancelled.")
            except Exception as e:
                print(f"Input error: {e}")
            print("!"*60 + "\n")

    def analyze_current_state(self):
        """分析当前状态并预测意图"""
        with self.prediction_lock:
            if self.latest_image is None:
                rospy.logwarn_throttle(5.0, 'No image received yet')
                return
            
            rospy.loginfo('Analyzing current state with LLaVA (Visual + Speech)...')
            
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
            
            # 发布预测结果
            intent_msg = String()
            intent_msg.data = response
            self.intent_pub.publish(intent_msg)
            
            rospy.loginfo(f'Predicted Intent:\n{response}\n{"-"*60}')
            
            # 交互式确认与执行
            self.check_and_prompt_action(response)
            
            # 发布调试图像
            if self.latest_image is not None:
                debug_image = self.latest_image.copy()
                # 添加简单的文本标注
                cv2.putText(debug_image, "VLM + Speech Mode", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if self.latest_speech:
                    cv2.putText(debug_image, f"Speech: {self.latest_speech}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
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

def main():
    try:
        predictor = IntentPredictorVLMSpeech()
        rospy.on_shutdown(predictor.cleanup)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
