import gym
import os
import cv2
import json
import zlib
import time
import rospy
import pickle
import base64
import numpy as np
from absl import app, flags
from datetime import datetime
from collections import OrderedDict
from flask import Flask, request, jsonify
from tiago_server.tiago_sys.tiago_ctrl.tiago_core import Tiago
from tiago_server.tiago_sys.tiago_ctrl.tiago_head import LookAtFixedPoint
from tiago_server.tiago_sys.utils.general_utils import AttrDict
from tiago_server.utils.flask_comm import encode2json, decode4json, serialize_space_dict


FLAGS = flags.FLAGS
flags.DEFINE_string("flask_url", "192.168.0.110", "URL for the flask server to run on.")
flags.DEFINE_string("flask_port", "1234", "Port for the flask server to run on.")


class TiagoEnv:
    def __init__(self,
                 frequency=10,
                 reset_pose={
                     'left': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
                     'right': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
                     'torso': 0.29,
                     'head': [0.0, -0.90],
                     },
                 all_act_keys={'left', 'right', 'base', 'torso'},
                 all_obs_keys = OrderedDict.fromkeys([
                     "left", "right", "left_joints", "right_joints",
                     "base_pose", "base_velocity", "torso",
                     "tiago_head_image",
                     ]),
                 ):
        
        self.frequency = frequency
        self.reset_pose = reset_pose
        self.all_act_keys = all_act_keys
        self.all_obs_keys = all_obs_keys
        
        ### initialize tiago core
        self.tiago = Tiago(
            head_policy=LookAtFixedPoint(reset_pose['head']),
            base_enabled=True,
            torso_enabled=True,
            left_arm_enabled=True,
            right_arm_enabled=True,
            right_gripper_type='pal',
            left_gripper_type='pal',
            reset_pose=reset_pose,
        )
        
        self.cameras = OrderedDict()
        self.cameras['tiago_head'] = self.tiago.head.head_camera
        
        self.steps = 0
        self.start_time = None
        
        ### reset tiago for execution
        self.obs = self.reset(reset_arms=True, is_input_cont=True)

    @property
    def state_space(self):
        st_space = OrderedDict()
        st_space['left'] = gym.spaces.Box(low=-1e10, high=1e10, shape=(8,), dtype=np.float32)
        st_space['right'] = gym.spaces.Box(low=-1e10, high=1e10, shape=(8,), dtype=np.float32)
        st_space['left_joints'] = gym.spaces.Box(low=-1e10, high=1e10, shape=(7,), dtype=np.float32)
        st_space['right_joints'] = gym.spaces.Box(low=-1e10, high=1e10, shape=(7,), dtype=np.float32)
        st_space['base_pose'] = gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32)
        st_space['base_velocity'] = gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32)
        st_space['torso'] = gym.spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32)
        for cam in self.cameras.keys():
            img_shape = tuple(self.cameras[cam].img_shape)
            depth_shape = tuple(self.cameras[cam].depth_shape)
            st_space[f'{cam}_image'] = gym.spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
            st_space[f'{cam}_depth'] = gym.spaces.Box(low=0, high=65535, shape=depth_shape, dtype=np.uint16)
        return gym.spaces.Dict(st_space)
    
    @property
    def observation_space(self):
        st_space = self.state_space.spaces
        ob_space = OrderedDict()
        for key in self.all_obs_keys:
            ob_space[key] = st_space[key]
        return gym.spaces.Dict(ob_space)
    
    @property
    def action_space(self):
        act_space = OrderedDict()
        act_space['left'] = gym.spaces.Box(low=-1e10, high=1e10, shape=(8,), dtype=np.float32)
        act_space['right'] = gym.spaces.Box(low=-1e10, high=1e10, shape=(8,), dtype=np.float32)
        act_space['base'] = gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32)
        act_space['torso'] = gym.spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32)
        return gym.spaces.Dict(act_space)

    def _state(self):
        def _get_arm_joints(side):
            joints = self.tiago.arms[side].joint_reader.get_most_recent_msg()
            return np.array(joints) if joints is not None else np.zeros(7)

        states = AttrDict({
            'left': np.r_[np.array(self.tiago.arms['left'].arm_pose), np.array(self.tiago.left_gripper_pos)],
            'right': np.r_[np.array(self.tiago.arms['right'].arm_pose), np.array(self.tiago.right_gripper_pos)],
            'left_joints': _get_arm_joints('left'),
            'right_joints': _get_arm_joints('right'),
            'base_pose': np.array(self.tiago.base.get_delta_pose()),
            'base_velocity': np.array(self.tiago.base.get_velocity()),
            'torso': np.array(self.tiago.torso.get_torso_extension()),
        })
        for cam in self.cameras.keys():
            states[f'{cam}_image'] = np.array(self.cameras[cam].get_img())
            states[f'{cam}_depth'] = np.array(self.cameras[cam].get_depth())
        return states

    def _state_wo_vis(self):
        def _get_arm_joints(side):
            joints = self.tiago.arms[side].joint_reader.get_most_recent_msg()
            return np.array(joints) if joints is not None else np.zeros(7)

        states = AttrDict({
            'left': np.r_[np.array(self.tiago.arms['left'].arm_pose), np.array(self.tiago.left_gripper_pos)],
            'right': np.r_[np.array(self.tiago.arms['right'].arm_pose), np.array(self.tiago.right_gripper_pos)],
            'left_joints': _get_arm_joints('left'),
            'right_joints': _get_arm_joints('right'),
            'base_pose': np.array(self.tiago.base.get_delta_pose()),
            'base_velocity': np.array(self.tiago.base.get_velocity()),
            'torso': np.array(self.tiago.torso.get_torso_extension()),
        })
        return states

    def _observation(self):
        states = self._state()
        observations = {k: states[k] for k in self.all_obs_keys if k in states}
        return observations

    def step(self, action):
        """
        Directly execute the action received from the Inference PC.
        """
        if action is not None:
            self.tiago.step(action)
            
        # Control loop timing
        end_time = time.time()
        if self.start_time is not None:
            rospy.sleep(max(0., 1/self.frequency - (end_time - self.start_time)))
        self.start_time = time.time()
        
        self.obs = self._observation()
        self.steps += 1
        return self.obs, {}
    
    def reset(self, *args, **kwargs):
        self.start_time = None
        self.steps = 0
        self.tiago.reset(*args, **kwargs)
        self.obs = self._observation()
        return self.obs
    
    def get_state(self):
        return self._state()
    
    def get_state_wo_vis(self):
        return self._state_wo_vis()
    
    def close(self):
        pass


def main(_):
    rospy.init_node("tiago_server_node")
    
    ### initialize the environment of Tiago
    tiago_server = TiagoEnv()

    ### build flask to communicate
    webapp = Flask(__name__)
    
    @webapp.route("/tiago_reset", methods=["POST"])
    def tiago_reset():
        reset_pose = decode4json(request.json["reset_pose"])
        obs = tiago_server.reset(reset_pose=reset_pose, reset_arms=True, is_input_cont=False)
        return jsonify(encode2json(obs))
    
    @webapp.route("/tiago_step", methods=["POST"])
    def tiago_step():
        action = decode4json(request.json["action"])
        obs, info = tiago_server.step(action)
        return jsonify({"obs": encode2json(obs), "info": encode2json(info)})
    
    @webapp.route("/tiago_get_state", methods=["POST"])
    def tiago_get_state():
        state = tiago_server.get_state()
        return jsonify(encode2json(state))
    
    @webapp.route("/tiago_get_state_wo_vis", methods=["POST"])
    def tiago_get_state_wo_vis():
        state_wo_vis = tiago_server.get_state_wo_vis()
        return jsonify(encode2json(state_wo_vis))
    
    @webapp.route("/tiago_get_action_space", methods=["POST"])
    def tiago_get_action_space():
        return jsonify(serialize_space_dict(tiago_server.action_space))
    
    @webapp.route("/tiago_get_observation_space", methods=["POST"])
    def tiago_get_observation_space():
        return jsonify(serialize_space_dict(tiago_server.observation_space))
    
    @webapp.route("/tiago_get_state_space", methods=["POST"])
    def tiago_get_state_space():
        return jsonify(serialize_space_dict(tiago_server.state_space))
    
    @webapp.route("/tiago_close", methods=["POST"])
    def tiago_close():
        tiago_server.close()
        return 'success'
    
    webapp.run(host=FLAGS.flask_url, 
               port=FLAGS.flask_port, 
               # debug=False, 
               debug=True, 
               use_reloader=False
               )


if __name__ == "__main__":
    app.run(main)
    


