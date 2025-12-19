import requests
import numpy as np
from collections import OrderedDict
from tiago_client.utils.flask_comm import decode4json, encode2json, reconstruct_space_dict

class TiagoClient:
    """
    Client interface for interacting with the TIAGO robot remotely.
    This class handles all HTTP communication with the Onboard PC.
    """
    def __init__(self, server_url="http://192.168.0.110:1234/"):
        self.url = server_url
        if not self.url.endswith("/"):
            self.url += "/"
            
        print(f"[TiagoClient] Connecting to {self.url}...")
        
        # Fetch robot specifications from the server
        self.action_space = self._obtain_action_space()
        self.observation_space = self._obtain_observation_space()
        self.state_space = self._obtain_state_space()
        
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

    def step(self, action, is_only_teleop=False):
        """
        Sends a control action to the robot.
        :param action: Dictionary of delta actions.
        :param is_only_teleop: If True, the robot will only follow teleop signals if present.
        :return: (observation, info)
        """
        action_json = encode2json(action)
        recept_json = requests.post(
            self.url + "tiago_step", 
            json={'action': action_json, 'is_only_teleop': is_only_teleop}
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
        Retrieves the current state of the Oculus VR controllers.
        Note: Requires the corresponding endpoint to be implemented on the Onboard PC.
        """
        try:
            response = requests.post(self.url + "tiago_get_oculus_state")
            if response.status_code == 200:
                return decode4json(response.json())
            else:
                print(f"[TiagoClient] Failed to get Oculus state: {response.status_code}")
                return None
        except Exception as e:
            print(f"[TiagoClient] Error calling get_oculus_state: {e}")
            return None

    def close(self):
        """Sends a signal to shut down the onboard server connection."""
        try:
            return requests.post(self.url + "tiago_close")
        except Exception:
            return None
