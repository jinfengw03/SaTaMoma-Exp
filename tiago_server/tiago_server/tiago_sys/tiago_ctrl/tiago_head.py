
import time
import rospy
import numpy as np
from pynput import keyboard
from std_msgs.msg import Header
from control_msgs.msg  import JointTrajectoryControllerState
from tiago_onboard.tiago_sys.utils.camera_utils import Camera
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tiago_onboard.tiago_sys.utils.ros_utils import Publisher, Listener


class TiagoHead:

    def __init__(self, head_policy) -> None:
        self.head_enabled = head_policy is not None
        self.head_policy = head_policy
        self.img_topic = "/xtion/rgb/image_raw"
        self.depth_topic = "/xtion/depth/image_raw"
        self.head_camera = Camera(img_topic=self.img_topic, depth_topic=self.depth_topic)
        self.setup_listener()
        self.setup_actors()

    def setup_listener(self):
        def process_head_state(message):
            return message.actual.positions
        self.head_listener = Listener(input_topic_name='/head_controller/state', input_message_type=JointTrajectoryControllerState, post_process_func=process_head_state)

    def setup_actors(self):
        self.head_writer = None
        if self.head_enabled:
            self.head_writer = Publisher('/head_controller/command', JointTrajectory)
    
    def create_head_command(self, pose):
        message = JointTrajectory()
        message.header = Header()
        message.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint(positions=pose, time_from_start=rospy.Duration(0.5))
        message.points.append(point)
        return message

    def write(self, pose):
        if self.head_enabled:
            self.head_writer.write(self.create_head_command(pose))
    
    def get_camera_obs(self):
        return self.head_camera.get_camera_obs()

    def step(self, env_action):
        pose = self.head_policy.get_action(env_action)
        if pose is None:
            return
        self.write(pose)

    def get_head_extension(self):
        current_head_extension = self.head_listener.get_most_recent_msg()
        return current_head_extension

    def reset(self, pose):
        if pose is None:
            return
        self.write(pose)


class TiagoHeadPolicy:

    def get_action(self, env_action, euler=True):
        '''
            transform env_action into [2d x + y]
        '''
        raise NotImplementedError


class FollowHandPolicy(TiagoHeadPolicy):

    def __init__(self, arm='right'):
        super().__init__()
        assert arm in ['right', 'left']
        self.arm = arm
    
    def get_action(self, env_action, euler=True):
        if env_action[self.arm] is None:
            return None
        pose = env_action[self.arm][:2] # need to update
        return pose


class LookAtFixedPoint(TiagoHeadPolicy):

    def __init__(self, pose) -> None:
        super().__init__()
        self.pose = pose
    
    def get_action(self, env_action, euler=True):
        pose = self.pose[:2]
        return pose


class TiagoHeadKeyboardTeleop:
    def __init__(self):
        self.controller = TiagoHead(head_policy=LookAtFixedPoint([0.0, 0.0]))
        self.delta_pose = np.zeros(2)  # [x, y]
        self.key_mapping = {
            # y \in [-0.98, 0.70]
            'w': (0.0, 0.1),   # up
            's': (0.0, -0.1),  # down
            # x \in [-1.23, 1.23]
            'a': (0.1, 0.0),   # left
            'd': (-0.1, 0.0),  # right
        }
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char in self.key_mapping:
                x, y = self.key_mapping[key.char]
                self.delta_pose = np.array([x, y])
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key.char in self.key_mapping:
                x, y = self.key_mapping[key.char]
                self.delta_pose -= np.array([x, y])
        except AttributeError:
            pass

    def run(self):
        print("begin to teleoperate...")
        rate = rospy.Rate(10)  # 10Hz loop
        while not rospy.is_shutdown():
            state = self.controller.get_head_extension()
            goal_pose = state + self.delta_pose
            print(state)
            self.controller.reset(goal_pose)
            rate.sleep()

    def look_at_fix_point(self, pose):
        print("begin to fix...")
        rate = rospy.Rate(10)  # 10Hz loop
        while not rospy.is_shutdown():
            self.controller.reset(pose)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node('tiago_head_check')
    head_tele = TiagoHeadKeyboardTeleop()
    rospy.sleep(1) # important for publisher to awake
    # calibrate the head
    head_tele.run()
    # # fix the head
    # head_tele.look_at_fix_point([0.0, 0.0])


