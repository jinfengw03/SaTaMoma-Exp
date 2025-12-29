#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Intent Predictor Node
意图预测节点
"""

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

class IntentPredictor:
    def __init__(self):
        rospy.init_node('intent_predictor_node')
        
        # 参数
        self.prediction_horizon = rospy.get_param('~prediction_horizon', 1.0)  # 预测时间范围（秒）
        self.update_rate = rospy.get_param('~update_rate', 10.0)  # 更新频率（Hz）
        
        # 订阅器
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback, queue_size=10)
        
        # 发布器
        self.intent_pub = rospy.Publisher('/predicted_intent', String, queue_size=10)
        self.target_pub = rospy.Publisher('/predicted_target', PoseStamped, queue_size=10)
        
        # 状态变量
        self.current_joint_positions = None
        self.joint_history = []
        self.max_history_length = 50
        
        rospy.loginfo('Intent Predictor Node initialized')
        
    def joint_callback(self, msg):
        """接收关节状态"""
        try:
            # 提取右臂关节角度
            arm_right_joints = ['arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint',
                               'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint',
                               'arm_right_7_joint']
            idx = [msg.name.index(j) for j in arm_right_joints]
            self.current_joint_positions = np.array([msg.position[i] for i in idx])
            
            # 保存历史数据
            self.joint_history.append(self.current_joint_positions.copy())
            if len(self.joint_history) > self.max_history_length:
                self.joint_history.pop(0)
                
        except (ValueError, IndexError) as e:
            rospy.logwarn_throttle(5.0, f'Could not extract joint positions: {e}')
    
    def predict_intent(self):
        """预测意图的主要方法"""
        if self.current_joint_positions is None:
            return
        
        if len(self.joint_history) < 5:
            return
        
        # 简单的速度估计
        velocity = self.joint_history[-1] - self.joint_history[-5]
        velocity_norm = np.linalg.norm(velocity)
        
        # 简单的意图分类
        if velocity_norm < 0.01:
            intent = "静止 (Stationary)"
        elif velocity_norm < 0.1:
            intent = "缓慢移动 (Slow Movement)"
        else:
            intent = "快速移动 (Fast Movement)"
        
        # 发布预测的意图
        intent_msg = String()
        intent_msg.data = intent
        self.intent_pub.publish(intent_msg)
        
        rospy.loginfo_throttle(2.0, f'Predicted Intent: {intent}, Velocity: {velocity_norm:.4f}')
    
    def run(self):
        """主循环"""
        rate = rospy.Rate(self.update_rate)
        while not rospy.is_shutdown():
            self.predict_intent()
            rate.sleep()

def main():
    try:
        predictor = IntentPredictor()
        predictor.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
