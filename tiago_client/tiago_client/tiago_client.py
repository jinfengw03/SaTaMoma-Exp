import os
import requests
import numpy as np
from collections import OrderedDict
from tiago_client.utils.flask_comm import decode4json, encode2json, reconstruct_space_dict
from tiago_client.oculus_teleop.teleop_policy import TeleopPolicy
from tiago_client.oculus_teleop.teleop_core import TeleopObservation
from tiago_client.utils.ik_solver import TiagoIK
from tiago_client.tiago_safety.safety_filter_right import JointSafetyFilter
from tiago_client.utils.transformations import quat_to_euler, euler_to_quat, add_angles

class TiagoClient:
    """
    Client interface for interacting with the TIAGO robot remotely.
    This class handles all HTTP communication with the Onboard PC
    and manages local teleoperation, IK, and safety filtering.
    """
    def __init__(self, server_url="http://192.168.0.110:1234/", use_teleop=True):
        self.url = server_url
        if not self.url.endswith("/"):
            self.url += "/"
            
        print(f"[TiagoClient] Connecting to {self.url}...")
        
        # Fetch robot specifications from the server
        self.action_space = self._obtain_action_space()
        self.observation_space = self._obtain_observation_space()
        self.state_space = self._obtain_state_space()
        
        # Initialize Teleop, IK, and Safety
        self.teleop = None
        if use_teleop:
            from tiago_client.tiago_client.oculus_teleop.configs.only_vr import teleop_config
            self.teleop = TeleopPolicy(teleop_config)
            self.teleop.start()
            
            # Initialize IK and Safety for both arms
            self.ik_solvers = {
                'right': TiagoIK(side='right'),
                'left': TiagoIK(side='left')
            }
            
            # Path to URDF (assuming it's in the same package)
            urdf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'urdf/tiago.urdf')
            self.safety_filters = {
                'right': JointSafetyFilter(urdf_path, side='right'),
                'left': JointSafetyFilter(urdf_path, side='left')
            }
        
        print("[TiagoClient] Connection established and spaces initialized.")

    def _obtain_state_space(self):
        data = requests.post(self.url + "tiago_get_state_space").json()
        return reconstruct_space_dict(data)
    
    def _obtain_observation_space(self):
        data = requests.post(self.url + "tiago_get_observation_space").json()
        return reconstruct_space_dict(data)
    
    def _obtain_action_space(self):
        data = requests.post(self.url + "tiago_get_action_space").json()
        return reconstruct_space_dict(data)

    def reset(self, reset_pose):
        """
        Resets the robot to a specified pose.
        :param reset_pose: Dictionary containing target positions for arms, base, torso, etc.
        """
        reset_pose_json = encode2json(reset_pose)
        recept_json = requests.post(self.url + "tiago_reset", json={'reset_pose': reset_pose_json}).json()
        return decode4json(recept_json)

    def step(self, action):
        """
        Sends a control action to the robot.
        :param action: Dictionary of actions (Cartesian or Joint).
        :return: (observation, info)
        """
        action_json = encode2json(action)
        recept_json = requests.post(
            self.url + "tiago_step", 
            json={'action': action_json}
        ).json()
        
        obs = decode4json(recept_json['obs'])
        info = decode4json(recept_json['info'])
        return obs, info

    def get_state(self):
        """Retrieves the full state of the robot including visual data."""
        data = requests.post(self.url + "tiago_get_state").json()
        return decode4json(data)
    
    def get_state_wo_vis(self):
        """Retrieves the robot state without heavy visual data (joints, poses only)."""
        data = requests.post(self.url + "tiago_get_state_wo_vis").json()
        return decode4json(data)

    def get_oculus_state(self):
        """
        Retrieves the current state of the Oculus VR controllers locally.
        """
        if self.teleop is not None:
            return self.teleop.interfaces['oculus'].get_state()
        return None

    def close(self):
        """Sends a signal to shut down the onboard server connection."""
        try:
            return requests.post(self.url + "tiago_close")
        except Exception:
            return None

    def get_teleop_action(self, is_filter=False, obstacles=None):
        """
        Reads the current VR controller input and calculates the robot action.
        :param is_filter: Whether to use the teleop policy's internal filter
        :param obstacles: List of [x, y, z, r] for the safety filter
        :return: (safe_action, buttons)
        """
        if self.teleop is None:
            return None, {}
            
        # Get current robot state
        state = self.get_state_wo_vis()
        
        # Prepare observation for teleop policy
        obs = TeleopObservation(
            left=state.get('left'),
            right=state.get('right'),
            base=state.get('base_pose'),
            torso=state.get('torso')[0] if isinstance(state.get('torso'), (list, np.ndarray)) else state.get('torso')
        )
        
        # Get raw Cartesian action from Oculus
        raw_action = self.teleop.get_action(obs, is_filter=is_filter)
        buttons = raw_action.extra.get('buttons', {})
        
        safe_action = {}
        
        # Process arms: Cartesian -> IK -> Safety Filter -> Joint Command
        for side in ['right', 'left']:
            if side in raw_action and raw_action[side] is not None:
                cartesian_delta = raw_action[side][:6]
                gripper_val = raw_action[side][6]
                
                # 1. Convert delta to absolute target pose
                cur_pose = state.get(side) # [x, y, z, qx, qy, qz, qw]
                cur_pos, cur_quat = cur_pose[:3], cur_pose[3:7]
                
                pos_delta, euler_delta = cartesian_delta[:3], cartesian_delta[3:6]
                cur_euler = quat_to_euler(cur_quat)
                target_pos = cur_pos + pos_delta
                target_euler = add_angles(euler_delta, cur_euler)
                target_quat = euler_to_quat(target_euler)
                
                # 2. Local IK
                joints_curr = state.get(f'{side}_joints')
                if joints_curr is not None:
                    joint_goal = self.ik_solvers[side].find_ik(target_pos, target_quat, joints_curr)
                    
                    if joint_goal is not None:
                        # 3. Safety Filter
                        if obstacles is not None:
                            self.safety_filters[side].update_obstacles(obstacles)
                        
                        joint_safe = self.safety_filters[side].filter(joints_curr, joint_goal)
                        
                        # 4. Combine with gripper (8 elements total)
                        safe_action[side] = np.concatenate([joint_safe, [gripper_val]])
                    else:
                        # If IK fails, stay at current joints
                        safe_action[side] = np.concatenate([joints_curr, [gripper_val]])
        
        # Process base and torso (direct pass-through for now)
        if 'base' in raw_action:
            safe_action['base'] = raw_action['base']
        if 'torso' in raw_action:
            safe_action['torso'] = raw_action['torso']
            
        return safe_action, buttons
