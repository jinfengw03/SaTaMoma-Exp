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
from std_msgs.msg import UInt8MultiArray, String
from importlib.machinery import SourceFileLoader
from tiago_onboard.tiago_sys.tiago_ctrl.tiago_gym import TiagoGym
from tiago_onboard.tiago_sys.tiago_ctrl.tiago_head import LookAtFixedPoint
from tiago_onboard.tiago_sys.oculus_teleop.teleop_policy import TeleopPolicy
from tiago_onboard.tiago_env.utils.flask_comm import encode2json, decode4json, serialize_space_dict


FLAGS = flags.FLAGS
flags.DEFINE_string("flask_url", "192.168.0.110", "URL for the flask server to run on.")
flags.DEFINE_string("flask_port", "1234", "Port for the flask server to run on.")


class TiagoEnv(TiagoGym):
    def __init__(self,
                 frequency=10, # not in use currently
                 teleop_config_path="/home/pal/jeffrey/RealWorldRL/real/workspace/hil_serl_onboard/tiago_onboard/tiago_onboard/tiago_sys/oculus_teleop/configs/only_vr.py",
                 reset_pose={
                     ### mobile-manip teleoperation status
                     # 'arm-joint': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
                     # 'arm-pose': [0.410, 0.311, 1.131, 0.015, 0.304, -0.362, 0.881, 1]
                     'left': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1], # reset through joints input | gripper=[abs-close(0)-open(1)]
                     'right': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1], # reset through joints input | gripper=[abs-close(0)-open(1)]
                     'torso': 0.29,
                     'head': [0.0, -0.90],
                     # ### default status
                     # 'right': [-1.10, 1.47, 2.71, 1.71, -1.57, 1.37, 0.00, 1],
                     # 'left': [-1.10, 1.47, 2.71, 1.71, -1.57, 1.37, 0.00, 1],
                     # 'torso': 0.20,
                     # 'head': [0.0, 0.0],
                     },
                 banned_act_keys={
                     ### left | right | base | torso
                     'torso',
                     }, 
                 all_act_keys={
                     'left',
                     'right',
                     'base',
                     'torso'
                     },
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
                 ):
        
        ### params initialization
        self.teleop_config = SourceFileLoader('conf', teleop_config_path).load_module().teleop_config
        self.reset_pose = reset_pose
        self.banned_act_keys = banned_act_keys
        self.all_act_keys = all_act_keys
        
        ### initialize tiago gym
        super().__init__(
            frequency=frequency,
            head_policy=LookAtFixedPoint(self.reset_pose['head']), # fixed eyeview
            base_enabled=(self.teleop_config.base_controller is not None) and ('base' not in self.banned_act_keys),
            torso_enabled=(self.teleop_config.base_controller is not None) and ('torso' not in self.banned_act_keys),
            left_arm_enabled=(self.teleop_config.arm_left_controller is not None) and ('left' not in self.banned_act_keys),
            right_arm_enabled=(self.teleop_config.arm_right_controller is not None) and ('right' not in self.banned_act_keys),
            right_gripper_type='pal',
            left_gripper_type='pal',
            reset_pose=self.reset_pose,
            all_obs_keys=all_obs_keys,
        )
        
        ### initialize teleoperation policy
        self.teleop_arm_triggers = {
            'left': 'LG',
            'right': 'RG',
            }
        self.teleop_threshold = {
            'base': 0.001,
            'torso': 0.001,
            }
        self.teleop = TeleopPolicy(self.teleop_config)
        self.teleop.start()

        ### reset tiago for execution
        self.obs = self.reset(reset_arms=True, is_input_cont=True)

    def retrieve_action(self, action):
        """
        @func: once teleoperating, fully take over the entire action space.
        """
        ### obtain teleoperation data
        teleop_signals = self.teleop.get_action(self.obs) # left | right | base | torso <-> all in delta value <-> gripper: 0(close) or 1(open)
        teleop_buttons = teleop_signals.extra.get('buttons', {})
        teleop_action = {}
        is_teleoped = False
        ### check teleoperation from human
        for key in self.all_act_keys:
            ## asign action from teleoperation
            teleop_action[key] = teleop_signals[key]
            ## check whether 
            if key not in self.banned_act_keys:
                if key in ['left', 'right']:
                    # arm trigger -> (x|y|z + rx|ry|rz) or (gripper)
                    if teleop_buttons[self.teleop_arm_triggers[key]]:
                        is_teleoped = True
                else:
                    # base or torso movement
                    if np.linalg.norm(teleop_action[key]) > self.teleop_threshold[key]:
                        is_teleoped = True
        ### returns once any teleoperation involved
        if is_teleoped:
            teleop_action = {k: teleop_action[k] for k in action.keys()} # filter unneeded actions
            return teleop_action, teleop_buttons, True
        else:
            return action, teleop_buttons, False
        
    def step(self, action, is_only_teleop=False):
        ### control by `teleoperation` only
        if is_only_teleop:
            teleop_signals = self.teleop.get_action(self.obs, is_filter=True) # left | right | base | torso <-> all in delta value <-> gripper: 0(close) or 1(open)
            teleop_buttons = teleop_signals.extra.get('buttons', {})
            exe_action_legal = {k: teleop_signals[k] for k in self.all_act_keys if k not in self.banned_act_keys}
            exe_action = {k: exe_action_legal[k] for k in action.keys()} # filter unneeded actions
            self.tiago.step(exe_action)
            self.obs = self._observation()
            self.steps += 1
            return self.obs, {'teleop_buttons': teleop_buttons, 'intervene_action': exe_action}
        ### control shared by `teleoperation` and `policy`
        else:
            exe_action, teleop_buttons, is_teleoped = self.retrieve_action(action) # [delta euler arm]*2 + [delta base]
            self.tiago.step(exe_action)
            self.obs = self._observation()
            info = {'teleop_buttons': teleop_buttons}
            if is_teleoped:
                info['intervene_action'] = exe_action
            self.steps += 1
            return self.obs, info
    
    def reset(self, *args, **kwargs):
        self.start_time = None
        self.end_time = None
        self.steps = 0
        self.tiago.reset(*args, **kwargs) # reset tiago -> `reset_arms` and `is_input_cont`
        self.teleop.interfaces['oculus'].reset_state() # reset teleoperation -> 
        self.obs = self._observation()
        return self.obs
    
    def get_state(self):
        return self._state()
    
    def get_state_wo_vis(self):
        return self._state_wo_vis()
    
    def close(self):
        self.teleop.stop()


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
        is_only_teleop = request.json["is_only_teleop"]
        obs, info = tiago_server.step(action, is_only_teleop)
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
    


