"""Gym Interface for TIAGO"""
import os
import cv2
import copy
import time
import queue
import requests
import threading
import numpy as np
from typing import Dict
import gymnasium as gym
from datetime import datetime
from collections import OrderedDict
from flask import Flask, request, jsonify
from scipy.spatial.transform import Rotation
from tiago_server.tiago_env.utils.flask_comm import decode4json, encode2json, reconstruct_space_dict


##############################################################################
class DefaultEnvConfig:
    """Default configuration for TiagoEnv. Fill in the values below."""
    
    ### basic initialization
    SERVER_URL: str = "http://192.168.0.110:1234/"  # fixed

    ### all poses and threshold
    
    # TARGET_POSE = {
    #     'left': np.zeros(((3+4+1),)),
    #     'right': np.zeros(((3+4+1),)),
    #     'base_pose': np.zeros(((3),)),
    #     'torso': np.zeros(((1),)),
    #     'head': np.zeros(((2),)),
    #     } # 22 = left-arm(8) + right-arm(8) + base(3) + torso(1) + head(2)
    TARGET_POSE = {
        # 'arm-joint': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
        # 'arm-pose': [0.410, 0.311, 1.131, 0.015, 0.304, -0.362, 0.881, 1]
        'left': np.array([0.410, 0.311, 1.131, 0.015, 0.304, -0.362, 0.881, 1]),     # x|y|z + x|y|z|w + grip [abs-close(0)-open(1)]
        'right': np.array([0.410, 0.311, 1.131, 0.015, 0.304, -0.362, 0.881, 1]),    # x|y|z + x|y|z|w + grip [abs-close(0)-open(1)]
        'base_pose': np.zeros(((3),)),                                               # [abs]
        'torso': np.array(0.29),                                                     # [abs]
        'head': np.array([0.0, -0.90]),                                              # [abs]
        } # 22 = left-arm(7) + right-arm(7) + base(3) + torso(1) + head(2)
    
    # RESET_POSE = {
    #     'left': np.zeros(((3+4+1),)),   # reset through joints input + grip [abs]
    #     'right': np.zeros(((3+4+1),)),  # reset through joints input + grip [abs]
    #     'base_pose': np.zeros(((3),)),  # [abs]
    #     'torso': np.zeros(((1),)),      # [abs]
    #     'head': np.zeros(((2),)),       # [abs]
    #     } # 22 = left-arm(8) + right-arm(8) + base(3) + torso(1) + head(2)
    RESET_POSE = {
        # 'arm-joint': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
        # 'arm-pose': [0.410, 0.311, 1.131, 0.015, 0.304, -0.362, 0.881, 1]
        'left': np.array([0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1]),           # reset through joints input + grip [abs-close(0)-open(1)]
        'right': np.array([0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1]),          # reset through joints input + grip [abs-close(0)-open(1)]
        'base_pose': np.zeros(((3),)),                                               # [abs]
        'torso': np.array(0.29),                                                     # [abs]
        'head': np.array([0.0, -0.90]),                                              # [abs]
        } # 22 = left-arm(7) + right-arm(7) + base(3) + torso(1) + head(2)
    
    REWARD_THRESHOLD = {
        'left': np.zeros((1,)),
        'right': np.zeros((1,)),
        'base_pose': np.zeros((1,)),
        # 'torso': np.zeros((1,)),
        # 'head': np.zeros((1,)),
    } # left-arm + right-arm + base + torso
    
    ### reset randomization
    RANDOM_RESET = False
    RANDOM_ARM_XY_RANGE = 0.0    # arms (left+right)
    RANDOM_ARM_RZ_RANGE = 0.0    # arms (left+right)
    RANDOM_BASE_XY_RANGE = 0.0   # base
    RANDOM_BASE_RZ_RANGE = 0.0   # base
    RANDOM_TORSO_RANGE = 0.0     # torso

    ### control the movement of TIAGO in a specific safety area <-> not necessary for the time being
    ABS_ARM_POSE_LIMIT_HIGH = np.zeros((3+3+1,))    # xyz | rpy | grip
    ABS_ARM_POSE_LIMIT_LOW = np.zeros((3+3+1,))     # xyz | rpy | grip
    ABS_BASE_LIMIT_HIGH = np.zeros((3,))            # x|y|rz
    ABS_BASE_LIMIT_LOW = np.zeros((3,))             # x|y|rz
    ABS_TORSO_LIMIT_LOW = np.zeros((1,))            # z
    ABS_TORSO_LIMIT_HIGH = np.zeros((1,))           # z

    ### other parameters
    MAX_EPISODE_LENGTH: int = 100
    # JOINT_RESET_PERIOD: int = 0 # not necessary for the time being
    # IMAGE_CROP: dict[str, callable] = {} # not necessary for the time being


##############################################################################
class TiagoEnv(gym.Env):
    """
    @notice: 
        a. all actions are [delta] + [euler] 
        b. all poses are [abs] + [quat]
    """
    def __init__(
        self,
        hz=10,          # execution frequency
        fake_env=False, # build a `ESC` listener for unfaked environment, vice versa
        config: DefaultEnvConfig = None,
    ):
        ### basic params
        self.config = config
        self.hz = hz
        self.url = config.SERVER_URL
        self._curr_traj_length = 0
        self._max_episode_length = config.MAX_EPISODE_LENGTH # environment trajectory length

        ### pre-defined poses
        self._reset_pose = config.RESET_POSE # left-arm(3+4+1) | right-arm(3+4+1) | base-movement(3) | torso(1) | head(2)
        self._target_pose = config.TARGET_POSE # left-arm(3+4+1) | right-arm(3+4+1) | base-movement(3) | torso(1) | head(2)
        self._reward_threshold = config.REWARD_THRESHOLD # left-arm | right-arm | base-movement | torso
        
        ### reset initialization
        self._is_random_reset = config.RANDOM_RESET
        self._random_arm_xy_range = config.RANDOM_ARM_XY_RANGE
        self._random_arm_rz_range = config.RANDOM_ARM_RZ_RANGE
        self._random_base_xy_range = config.RANDOM_BASE_XY_RANGE
        self._random_base_rz_range = config.RANDOM_BASE_RZ_RANGE
        self._random_torso_range = config.RANDOM_TORSO_RANGE
        
        ### set self-design action bounding for safety
        self.safety_bounding_boxes = {
            'left': gym.spaces.Box(
                config.ABS_ARM_POSE_LIMIT_LOW,
                config.ABS_ARM_POSE_LIMIT_HIGH,
                dtype=np.float32,
                ), # x|y|z|rx|ry|rz|gripper
            'right': gym.spaces.Box(
                config.ABS_ARM_POSE_LIMIT_LOW,
                config.ABS_ARM_POSE_LIMIT_HIGH,
                dtype=np.float32,
                ), # x|y|z|rx|ry|rz|gripper
            'base': gym.spaces.Box(
                config.ABS_BASE_LIMIT_LOW,
                config.ABS_BASE_LIMIT_HIGH,
                dtype=np.float32,
                ), # x|y|rz
            # 'torso': gym.spaces.Box(
            #     config.ABS_TORSO_LIMIT_LOW,
            #     config.ABS_TORSO_LIMIT_HIGH,
            #     dtype=np.float32,
            #     ), # z
        }
        
        ### action space from onboard
        self.action_space = self._obtain_action_space()

        ### observation space from onboard
        self.observation_space = self._obtain_observation_space()
        
        ### state space
        self.state_space = self._obtain_state_space()
        
        ### if fake_env there's no need for keyboard listener
        if not fake_env:
            from pynput import keyboard
            self.terminate = False
            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
        
        print("[TiagoEnv] Have initialized Tiago")
    
    def get_max_episode_length(self):
        return self._max_episode_length
    
    def _obtain_state_space(self):
        data = requests.post(self.url + "tiago_get_state_space").json()
        return reconstruct_space_dict(data)
    
    def _obtain_observation_space(self):
        data = requests.post(self.url + "tiago_get_observation_space").json()
        return reconstruct_space_dict(data)
    
    def _obtain_action_space(self):
        data = requests.post(self.url + "tiago_get_action_space").json()
        return reconstruct_space_dict(data)
    
    def _action_clip_safety_bound(self, action):
        """
        @func: clip the absolute action to be within the safety box.
        @notice: 
        """
        ### left-arm <-> x|y|z|rx|ry|rz|gripper // right-arm <-> x|y|z|rx|ry|rz|gripper // base <-> x|y|rz // torso <-> z
        for key in action.keys():
            action[key] = np.clip(action[key], self.safety_bounding_boxes[key].low, self.safety_bounding_boxes[key].high)    
        ### return the action
        return action
    
    def _action_clip_space_bound(self, action):
        """
        @func: clip the delta action to be within the action space bounding box.
        """
        ### left-arm <-> x|y|z|rx|ry|rz|gripper // right-arm <-> x|y|z|rx|ry|rz|gripper // base <-> x|y|rz // torso <-> z
        for key in action.keys():
            action[key] = np.clip(action[key], self.action_space[key].low, self.action_space[key].high)
        ### return the action
        return action
    
    def get_state(self):
        data = requests.post(self.url + "tiago_get_state").json()
        return decode4json(data)
    
    def get_state_wo_vis(self):
        data = requests.post(self.url + "tiago_get_state_wo_vis").json()
        return decode4json(data)
    
    def step(self, action, is_only_teleop=False) -> tuple:
        """
        @func: standard gym step function.
        @notice: all actions are delta
        """
        start_time = time.time() # start time
        action = self._action_clip_space_bound(action) # clip through space bounding
        # action = self._action_clip_safety_bound(action) # clip through safety bounding
        action_json = encode2json(action)
        recept_json = requests.post(self.url + "tiago_step", json={'action': action_json, 'is_only_teleop': is_only_teleop}).json()
        obs = decode4json(recept_json['obs'])
        info = decode4json(recept_json['info'])
        self._curr_traj_length += 1
        duration_time = time.time() - start_time # env time to maintain frequency
        # print('[T] ', duration_time) # for test
        time.sleep(max(0, (1.0 / self.hz) - duration_time)) # to maintain the execution frequency
        reward = int(self.compute_reward(self.get_state_wo_vis())) # calculate through real-world states
        done = self._curr_traj_length >= self._max_episode_length or reward or self.terminate
        info['succeed'] = reward
        truncated = False # Not in use temporarily
        return obs, reward, done, truncated, info
    
    def compute_reward(self, state_wo_vis) -> bool:
        """
        @func: compute the rewards based on states | rigid prediction
        """
        ### calculate the delta distance
        is_succeed = True
        for key, value in self._reward_threshold.items():
            current_pose = state_wo_vis[key] # arm -> [abs] + [quat] // base,torso,head -> [abs]
            target_pose = self._target_pose[key] # arm -> [abs] + [quat] // base,torso,head -> [abs]
            ### calculate delta
            if key in ['left', 'right']: # left, right arms
                current_rot = Rotation.from_quat(current_pose[3:7]).as_matrix() # quat to matrix
                target_rot = Rotation.from_quat(target_pose[3:7]).as_matrix() # quat to matrix
                diff_rot = current_rot.T @ target_rot
                diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
                delta = np.abs(np.hstack([current_pose[:3] - target_pose[:3], diff_euler])) # without consideration of gripper
            else: # base, torso, head
                delta = np.abs(current_pose - target_pose)
            ### check whether succeed through state distance
            if np.sum(delta) > value:
                is_succeed = False
                break
        ### judge whether success
        if is_succeed:
            # print(f'Goal reached, the difference is {delta}, the desired threshold is {self._reward_threshold}')
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._reward_threshold}')
            return False
    
    def reset(self, **kwargs):
        """
        @func: reset the tiago robot with randomization
        """
        reset_pose = self._reset_pose.copy()
        if self._is_random_reset:
            # arms
            for key in ['left', 'right']:
                reset_pose[key][:2] += np.random.uniform(-self._random_arm_xy_range, self._random_arm_xy_range, (2,))   # xy plane
                reset_pose[key][5:6] += np.random.uniform(-self._random_arm_rz_range, self._random_arm_rz_range)        # rz plane
            # base
            reset_pose['base_pose'][:2] += np.random.uniform(-self._random_base_xy_range, self._random_base_xy_range, (2,))  # xy plane
            reset_pose['base_pose'][2:3] += np.random.uniform(-self._random_base_rz_range, self._random_base_rz_range)       # rz plane
            # torso
            reset_pose['torso'] += np.random.uniform(-self._random_torso_range, self._random_torso_range)               # z plane
        reset_pose_json = encode2json(reset_pose)
        recept_json = requests.post(self.url + "tiago_reset", json={'reset_pose': reset_pose_json}).json()
        obs = decode4json(recept_json)
        self._curr_traj_length = 0
        self.terminate = False
        info = {"succeed": False}
        return obs, info
    
    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        result = requests.post(self.url + "tiago_close")    
        return result

##############################################################################
def static_test(env):
    ### 
    print("--------------------------------")
    print(env.action_space)
    print(env.observation_space)
    print(env.state_space)
    ### 
    state = env.get_state()
    state_wo_vis = env.get_state_wo_vis()
    print("--------------------------------")
    print(state.keys())
    print("--------------------------------")
    print(state_wo_vis.keys())
    time.sleep(3)
    ### 
    action = {
        'left': np.array([0,0,0,0,0,0,1]), # open gripper (1)
        'right': np.array([0,0,0,0,0,0,0]), # close gripper (0)
        'base': np.zeros((3,)),
        # 'torso': np.zeros((1,)),
    }
    obs, reward, done, truncated, info = env.step(action)
    print("--------------------------------")
    print(obs)
    print("--------------------------------")
    print(reward)
    print("--------------------------------")
    print(done)
    print("--------------------------------")
    print(truncated)
    print("--------------------------------")
    print(info)
    time.sleep(3)
    ### 
    obs, info = env.reset()
    print("--------------------------------")
    print(obs)
    print("--------------------------------")
    print(info)

def continuous_test(env):
    while True:
        action = OrderedDict(
            (k, np.zeros(space.shape, dtype=space.dtype))
            for k, space in env.action_space.spaces.items()
        ) # all zero action <-> only collect human interaction
        for k in ("left", "right"):
            if k in action:
                action[k][-1] = 1 # open the gripper as default
        # next_obs, rew, done, truncated, info = env.step(action) # environment interaction
        next_obs, rew, done, truncated, info = env.step(action, is_only_teleop=True) # environment interaction w/ only teleoperation
        if "intervene_action" in info: # human intervene
            action = info["intervene_action"] # assign human intervention data
            print("[HUMAN] ", action['right'])
        else:
            print("[-----] ", action['right'])

if __name__ == "__main__":
    
    ### build the connection to onboard environment
    env = TiagoEnv(fake_env=False, 
                   config=DefaultEnvConfig())
    
    # ### test through static command
    # static_test(env)

    ### reset the tiago    
    obs, info = env.reset()

    ### test through continuous commands
    continuous_test(env)
