import gym
import time
import rospy
import numpy as np
from collections import OrderedDict
from tiago_onboard.tiago_sys.tiago_ctrl.tiago_core import Tiago
from tiago_onboard.tiago_sys.utils.general_utils import AttrDict


class TiagoGym(gym.Env):

    def __init__(
            self,
            frequency=10,
            head_policy=None,
            base_enabled=False,
            torso_enabled=False,
            right_arm_enabled=True,
            left_arm_enabled=True,
            right_gripper_type=None,
            left_gripper_type=None,
            reset_pose=None,
            all_obs_keys = OrderedDict.fromkeys([
                "left",
                "right",
                "base_pose",
                "base_velocity",
                "torso",
                # "scan",
                "tiago_head_image",
                # "tiago_head_depth",
            ]),
            external_cams={}, # not in use currently
            ):
        super(TiagoGym).__init__()
        self.frequency = frequency
        self.base_enabled = base_enabled
        self.torso_enabled = torso_enabled
        self.right_arm_enabled = right_arm_enabled
        self.left_arm_enabled = left_arm_enabled
        self.right_gripper_enabled = right_gripper_type is not None
        self.left_gripper_enabled = left_gripper_type is not None
        self.tiago = Tiago(
                        head_policy=head_policy,
                        base_enabled=base_enabled,
                        torso_enabled=torso_enabled,
                        right_arm_enabled=right_arm_enabled,
                        left_arm_enabled=left_arm_enabled,
                        right_gripper_type=right_gripper_type,
                        left_gripper_type=left_gripper_type,
                        reset_pose=reset_pose,
                    )
        self.cameras = OrderedDict()
        self.cameras['tiago_head'] = self.tiago.head.head_camera
        for cam_name in external_cams.keys():
            self.cameras[cam_name] = external_cams[cam_name]
        self.all_obs_keys = all_obs_keys
        self.steps = 0
    
    @property
    def state_space(self):
        st_space = OrderedDict()
        st_space['left'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3+4+1,), # [3d position + 4d quarternion orientation + 1d gripper] | gripper=[abs-close(0)-open(1)]
            dtype=np.float32
        )
        st_space['right'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3+4+1,), # [3d position + 4d quarternion orientation + 1d gripper] | gripper=[abs-close(0)-open(1)]
            dtype=np.float32
        )
        st_space['base_pose'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2+1,), # [2d x, y position delta + 1d z orientation delta]
            dtype=np.float32
        )
        st_space['base_velocity'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2+1,), # [2d x, y linear velocity + 1d z angular velocity]
            dtype=np.float32
        )
        st_space['torso'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,), # [1d z]
            dtype=np.float32
        )
        # st_space['scan'] = gym.spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(572,), # [572d laser range finder] <-> TODO: figure out the usage of it
        #     dtype=np.float32
        # )
        for cam in self.cameras.keys():
            st_space[f'{cam}_image'] = gym.spaces.Box(
                low=0,      # -np.inf,
                high=255,   # np.inf,
                shape=self.cameras[cam].img_shape,
                dtype=np.uint8
            )
            st_space[f'{cam}_depth'] = gym.spaces.Box(
                low=0,      # -np.inf,
                high=65535, # np.inf,
                shape=self.cameras[cam].depth_shape,
                dtype=np.uint16
            )
        return gym.spaces.Dict(st_space)
    
    @property
    def observation_space(self):
        st_space = self.state_space
        ob_space = OrderedDict()
        for key in self.all_obs_keys:
            if key not in st_space.spaces.keys():
                raise KeyError(f"Observation key '{key}' not found in state_space.")
            ob_space[key] = st_space.spaces[key]
        return gym.spaces.Dict(ob_space)
    
    @property
    def action_space(self):
        act_space = OrderedDict()
        if self.left_arm_enabled:
            act_space['left'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3+3+int(self.left_gripper_enabled),), # [3d position + 3d roll-pitch-yaw orientation + 1d gripper]
                dtype=np.float32,
            )
        if self.right_arm_enabled:
            act_space['right'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3+3+int(self.right_gripper_enabled),), # [3d position + 3d roll-pitch-yaw orientation + 1d gripper]
                dtype=np.float32,
            )
        if self.base_enabled:
            act_space['base'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,), # [2d x + y linear velocity + 1d z angular velocity]
                dtype=np.float32,
            )
        if self.torso_enabled:
            act_space['torso'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,), # [1d up and downs]
                dtype=np.float32,
            )
        return gym.spaces.Dict(act_space)
    
    def _state(self):
        states = AttrDict({
            'left': np.r_[np.array(self.tiago.arms['left'].arm_pose), np.array(self.tiago.left_gripper_pos)],       # float64 -> float32
            'right': np.r_[np.array(self.tiago.arms['right'].arm_pose), np.array(self.tiago.right_gripper_pos)],    # float64 -> float32
            'base_pose': np.array(self.tiago.base.get_delta_pose()),            # float64 -> float32 -> based on origin recorded by odometry
            'base_velocity': np.array(self.tiago.base.get_velocity()),          # float64 -> float32 -> velocity at the point
            'torso': np.array(self.tiago.torso.get_torso_extension()),          # float64 -> float32
            # 'scan': np.array(self.tiago.base.get_scan())                        # float64 -> float32
        })
        for cam in self.cameras.keys():
            states[f'{cam}_image'] = np.array(self.cameras[cam].get_img())      # uint64 -> uint8 | [0-255]
            states[f'{cam}_depth'] = np.array(self.cameras[cam].get_depth())    # uint64 -> uint16 | [0-65535]
        return states
    
    def _observation(self):
        states = self._state()
        observations = {k: states[k] for k in self.all_obs_keys}
        return observations
    
    def _state_wo_vis(self):
        states = AttrDict({
            'left': np.r_[np.array(self.tiago.arms['left'].arm_pose), np.array(self.tiago.left_gripper_pos)], # xyz + xyzw + gripper
            'right': np.r_[np.array(self.tiago.arms['right'].arm_pose), np.array(self.tiago.right_gripper_pos)], # xyz + xyzw + gripper
            'base_pose': np.array(self.tiago.base.get_delta_pose()), # movement current pose -> based on origin recorded by odometry
            'base_velocity': np.array(self.tiago.base.get_velocity()), # movement current velocity
            'torso': np.array(self.tiago.torso.get_torso_extension()),
            # 'scan': np.array(self.tiago.base.get_scan())
        })
        return states
    
    def reset(self, *args, **kwargs):
        self.start_time = None
        self.end_time = None
        self.steps = 0
        self.tiago.reset(*args, **kwargs)
        return self._observation()
    
    def step(self, action):
        if action is not None:
            self.tiago.step(action)
        self.end_time = time.time()
        if self.start_time is not None:
            # print('Idle time:', 1/self.frequency - (self.end_time-self.start_time))
            rospy.sleep(max(0., 1/self.frequency - (self.end_time-self.start_time)))
        self.start_time = time.time()
        obs = self._observation()
        rew = 0
        done = False
        info = {}
        self.steps += 1
        return obs, rew, done, info