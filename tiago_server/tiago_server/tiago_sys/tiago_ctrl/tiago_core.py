import time
import rospy
from tiago_server.tiago_sys.tiago_ctrl.tiago_head import TiagoHead
from tiago_server.tiago_sys.tiago_ctrl.tiago_arms import TiagoArms
from tiago_server.tiago_sys.tiago_ctrl.tiago_torso import TiagoTorso
from tiago_server.tiago_sys.tiago_ctrl.tiago_head import LookAtFixedPoint
from tiago_server.tiago_sys.tiago_ctrl.tiago_mobile_base import TiagoBaseVelocityControl
from tiago_server.tiago_sys.tiago_ctrl.grippers import PALGripper, RobotiqGripper2F_140, RobotiqGripper2F_85


class Tiago:
    gripper_map = {'pal': PALGripper, 'robotiq2F-140': RobotiqGripper2F_140, 'robotiq2F-85': RobotiqGripper2F_85}
    
    def __init__(
        self,
        head_policy=None,
        base_enabled=False,
        torso_enabled=False,
        right_arm_enabled=True,
        left_arm_enabled=True,
        right_gripper_type=None,
        left_gripper_type=None,
        reset_pose=None,
    ):
        # params
        self.head_enabled = head_policy is not None
        self.base_enabled = base_enabled
        self.torso_enabled = torso_enabled
        self.head = TiagoHead(head_policy=head_policy)
        self.base = TiagoBaseVelocityControl(base_enabled=base_enabled)
        self.torso = TiagoTorso(torso_enabled=torso_enabled)
        self.arms = {
            'right': TiagoArms(right_arm_enabled, side='right'),
            'left': TiagoArms(left_arm_enabled, side='left'),
        }
        # set up gripper | gripper=[abs-close(0)-open(1)]
        self.gripper = {'right': None, 'left': None}
        for side in ['right', 'left']:
            gripper_type = right_gripper_type if side=='right' else left_gripper_type
            if gripper_type is not None:
                self.gripper[side] = self.gripper_map[gripper_type](side)
        # set reset pose
        if reset_pose is None:
            self.reset_pose = {
                    'right': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
                    'left': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
                    'torso': 0.15,
                    'head': [0.0, 0.0],
                }
        else:
            self.reset_pose = reset_pose
    
    @property
    def right_gripper_pos(self):
        if self.gripper['right'] is None:
            return None
        return self.gripper['right'].get_state()
    
    @property
    def left_gripper_pos(self):
        if self.gripper['left'] is None:
            return None
        return self.gripper['left'].get_state()
    
    def step(self, action):
        # torso 
        if self.torso_enabled and (self.torso is not None):
            if action.get('torso') is not None:
                self.torso.step(action['torso']) # delta
        # arm 
        for side in ['right', 'left']:
            if action.get(side) is None:
                continue
            
            if len(action[side]) == 8: # joint control: 7 joints + 1 gripper
                joint_action = action[side][:7]
                gripper_action = action[side][7]
                self.arms[side].step_joints(joint_action)
                if self.gripper[side] is not None:
                    self.gripper[side].step(gripper_action)
            else: # cartesian control: 6 delta + 1 gripper
                arm_action = action[side][:6]
                gripper_action = action[side][6]
                self.arms[side].step(arm_action) # position delta + euler angle
                if self.gripper[side] is not None:
                    self.gripper[side].step(gripper_action) # abs 
        # head 
        if self.head_enabled:
            self.head.step(action) # TBD: ...
        # base 
        if self.base_enabled:
            if action.get('base') is not None:
                self.base.step(action['base']) # delta
    
    def reset(self, reset_pose=None, reset_arms=True, is_input_cont=False):
        # choose the pose to reset
        if reset_pose is not None:
            exe_reset_pose = reset_pose
        else:
            exe_reset_pose = self.reset_pose
        # torso
        if ('torso' in exe_reset_pose.keys()) and (self.torso is not None):
            print(f'resetting torso...{time.time()}')
            self.torso.reset(exe_reset_pose['torso']) # [abs]
            rospy.sleep(3)
        # arms
        for side in ['right', 'left']:
            if (exe_reset_pose[side] is not None) and (self.arms[side].arm_enabled):
                self.gripper[side].step(exe_reset_pose[side][-1]) # [abs]
                if reset_arms:
                    print(f'resetting {side}...{time.time()}')
                    self.arms[side].reset(exe_reset_pose[side][:-1]) # [abs] + quat
                    rospy.sleep(1)
        # head
        if self.head_enabled:
            print(f'resetting head...{time.time()}')
            self.head.reset(exe_reset_pose['head']) # [abs]
        # base
        # ... if ('base_pose' in exe_reset_pose.keys()) and (self.base is not None) ...
        
        # sleep to wait for initialization
        rospy.sleep(0.5)
        # whether to interact
        if is_input_cont:
            input('Reset complete. Press ENTER to continue')


if __name__ == "__main__":
    rospy.init_node('tiago_core_check')
    tiago = Tiago(
                head_policy=LookAtFixedPoint([0.0, 0.0]),
                base_enabled=True,
                torso_enabled=True,
                right_arm_enabled=True,
                left_arm_enabled=True,
                right_gripper_type='pal',
                left_gripper_type='pal'
            )
    rospy.sleep(1) # important for publisher to awake
    tiago.reset()