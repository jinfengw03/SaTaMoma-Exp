import os
import time
from typing import Dict
import numpy as np
from tiago_client.utils.general_utils import run_threaded_command
from tiago_client.utils.transformations import quat_diff, quat_to_euler, rmat_to_quat
from tiago_client.oculus_teleop.teleop_core import BaseTeleopInterface, TeleopAction, TeleopObservation


def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X

#### code adapted from https://github.com/AlexanderKhazatsky/R2D2/blob/main/r2d2/controllers/oculus_controller.py ####
class OculusPolicy(BaseTeleopInterface):
    def __init__(
        self,
        max_lin_vel: float = 1, # the maximum linear velocity
        max_rot_vel: float = 1, # the maximum rotate velocity
        max_gripper_vel: float = 1,
        spatial_coeff: float = 1,
        pos_action_gain: float = 5,
        rot_action_gain: float = 2,
        gripper_action_gain: float = 3,
        rmat_reorder: list = [-2, -1, -3, 4],
        *args,
        **kwargs
    ) -> None:
        # lazy import so that we can use the rest of the code without installing oculus_reader
        try:
            import oculus_reader
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "oculus_reader not installed! Please visit https://github.com/rail-berkeley/oculus_reader for installation instrucitons."
            )
        super().__init__(*args, **kwargs)
        from oculus_reader.oculus_reader.reader import OculusReader

        # Prefer explicit kwargs, then env vars, else fall back to None (USB like reader.py)
        ip_cfg = kwargs.get('ip_address', None)
        port_cfg = kwargs.get('port', None)
        ip_env = os.getenv('OCULUS_IP', None)
        port_env = os.getenv('OCULUS_PORT', None)
        ip_address = ip_cfg if ip_cfg is not None else ip_env
        port = int(port_cfg if port_cfg is not None else (port_env if port_env is not None else 5555))

        # If ip_address is None, OculusReader will behave like reader.py (USB)
        self.oculus_reader = OculusReader(run=False, ip_address=ip_address, port=port)
        
        self.vr_to_global_mat = {'right': np.eye(4), 'left': np.eye(4)}
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)
        self.reset_orientation = {'right': True, 'left': True}
        self.target_gripper = {'right': 1, 'left': 1}
        self.reset_state()
        
    def start(self) -> None:
        self.oculus_reader.run()
        run_threaded_command(self._update_internal_state) # run a sub-thread for oculus

    def stop(self) -> None:
        print("Stopping Oculus Interface...")
        self.oculus_reader.stop() # close a sub-thread for oculus

    def reset_state(self) -> None:
        self._state = {
            'right': {
                "poses": None,
                "movement_enabled": False,
                "controller_on": True,
                "prev_gripper": False,
                "gripper_toggle": False,
            },

            'left': {
                "poses": None,
                "movement_enabled": False,
                "controller_on": True,
                "prev_gripper": False,
                "gripper_toggle": False,
            },
            'buttons': {}
        }
        self.update_sensor = {'right': True, 'left': True}
        self.reset_origin = {'right': True, 'left': True}
        self.robot_origin = {'right': None, 'left': None}
        self.vr_origin = {'right': None, 'left': None}
        self.vr_state = {'right': None, 'left': None}

    def _update_internal_state(self, num_wait_sec=5, hz=50):
        last_read_time = time.time()
        while True:

            # regulate read frequency
            time.sleep(1 / hz)

            # read controller
            time_since_read = time.time() - last_read_time
            poses, buttons = self.oculus_reader.get_transformations_and_buttons() # get pose and buttons
            if poses == {}:
                # print('skip')
                continue

            # determine control pipeline
            for arm in ['left', 'right']:
                button_G = 'RG' if arm=='right' else 'LG' # only update when button_G is pressed
                button_J = 'RJ' if arm=='right' else 'LJ'
                controller_id = 'r' if arm=='right' else 'l'

                # assign according to buttons
                if controller_id not in poses:
                    continue
                self._state[arm]["controller_on"] = time_since_read < num_wait_sec
                toggled = self._state[arm]["movement_enabled"] != buttons[button_G] # only toggled by button_G (arms)
                self.update_sensor[arm] = self.update_sensor[arm] or buttons[button_G]
                self.reset_orientation[arm] = self.reset_orientation[arm] or buttons[button_J]
                self.reset_origin[arm] = self.reset_origin[arm] or toggled # only consider arms

                # save pose info
                self._state[arm]["poses"] = poses[controller_id]
                self._state["buttons"] = buttons
                self._state[arm]["movement_enabled"] = buttons[button_G]
                self._state[arm]["controller_on"] = True

                # save gripper info
                new_gripper = buttons[f"{arm}Trig"][0] > 0.5
                self._state[arm]["gripper_toggle"] = ((not self._state[arm]["prev_gripper"]) and new_gripper) or self._state[arm]["gripper_toggle"]
                self._state[arm]["prev_gripper"] = new_gripper

                last_read_time = time.time()

                # update reset
                stop_updating = self._state["buttons"][button_J] or self._state[arm]["movement_enabled"]
                if self.reset_orientation[arm]:
                    rot_mat = np.asarray(self._state[arm]["poses"])
                    if stop_updating:
                        self.reset_orientation[arm] = False
                    # try to invert the rotation matrix, if not possible, then just use the identity matrix                
                    try:
                        rot_mat = np.linalg.inv(rot_mat)
                    except:
                        print(f"exception for rot mat: {rot_mat}")
                        rot_mat = np.eye(4)
                        self.reset_orientation[arm] = True
                    self.vr_to_global_mat[arm] = rot_mat
    
    def _process_reading(self, arm):
        rot_mat = np.asarray(self._state[arm]["poses"])
        rot_mat = self.global_to_env_mat @ self.vr_to_global_mat[arm] @ rot_mat
        vr_pos = self.spatial_coeff * rot_mat[:3, 3]
        vr_quat = rmat_to_quat(rot_mat[:3, :3])
        vr_gripper = self._state["buttons"]["rightTrig"][0] if arm=='right' else self._state["buttons"]["leftTrig"][0]
        gripper_toggle = self._state[arm]["gripper_toggle"]
        self._state[arm]["gripper_toggle"] = False
        self.vr_state[arm] = {"pos": vr_pos, "quat": vr_quat, "gripper": vr_gripper, "gripper_toggle": gripper_toggle}
    
    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        gripper_vel_norm = np.linalg.norm(gripper_vel)
        if lin_vel_norm > self.max_lin_vel:
            lin_vel = lin_vel * self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel:
            rot_vel = rot_vel * self.max_rot_vel / rot_vel_norm
        if gripper_vel_norm > self.max_gripper_vel:
            gripper_vel = gripper_vel * self.max_gripper_vel / gripper_vel_norm
        return lin_vel, rot_vel, gripper_vel

    def _calculate_action(self, robot_obs: Dict[str, np.ndarray], arm: str, is_filter=False) -> np.ndarray:

        # read sensor
        if self.update_sensor[arm]:
            self._process_reading(arm)
            self.update_sensor[arm] = False
        
        # read observation
        robot_pos = np.array(robot_obs["cartesian_position"][:3])
        robot_quat = robot_obs["cartesian_position"][3:]
        
        # reset origin on release
        if self.reset_origin[arm]:
            self.robot_origin[arm] = {"pos": robot_pos, "quat": robot_quat} # original robot state
            self.vr_origin[arm] = {"pos": self.vr_state[arm]["pos"], "quat": self.vr_state[arm]["quat"]} # original VR state
            self.reset_origin[arm] = False
        
        # calculate positional action (offset differnce to oculus) <-> according to the original recorded action [pos]
        target_pos_offset = self.vr_state[arm]["pos"] - self.vr_origin[arm]["pos"]
        if is_filter:
            robot_pos_offset = robot_pos - self.robot_origin[arm]["pos"]
            pos_action = target_pos_offset - robot_pos_offset # compensate for robot's deviation
        else:
            pos_action = target_pos_offset
        
        # calculate euler action (offset differnce to oculus) <-> according to the original recorded action [quat]
        target_quat_offset = quat_diff(self.vr_state[arm]["quat"], self.vr_origin[arm]["quat"])
        if is_filter:
            robot_quat_offset = quat_diff(robot_quat, self.robot_origin[arm]["quat"]) 
            quat_action = quat_diff(target_quat_offset, robot_quat_offset) # compensate for robot's deviation
            euler_action = quat_to_euler(quat_action) 
        else:
            euler_action = quat_to_euler(target_quat_offset)
        
        # robot movement according to human movement
        delta_action = np.concatenate((pos_action, euler_action)) # x|y|z + rx|ry|rz
        
        # gripper value map directly <-> updated when gripper toggle
        if self.vr_state[arm]["gripper_toggle"]:
            self.target_gripper[arm] = 1 - int(robot_obs["gripper_position"] > 0.5) # gripper target <-> {0,1}/{close,open} <-> instruct reversal
        
        # prepare return values
        action = np.concatenate([delta_action, [self.target_gripper[arm]]])
        action = action.clip(-1, 1)
        
        return action

    def get_action(self, obs: TeleopObservation, is_filter=False) -> TeleopAction:

        action = self.get_default_action()
        
        # add button data to action
        buttons = self._state["buttons"]
        action.extra['buttons'] = buttons

        # arm command
        for arm in ['right', 'left']:
            eef_data = obs[arm]
            if eef_data is None:
                continue
            # Use full 7-element pose (x,y,z,qx,qy,qz,qw); gripper from VR buttons, not this pose.
            robot_obs = {'cartesian_position': eef_data[:7], 'gripper_position': 0.0}
            if self._state[arm]["poses"] is not None:
                action[arm] = self._calculate_action(robot_obs, arm, is_filter)

        # base command
        base_action = np.zeros(3)
        if 'rightJS' in buttons:
            base_action[0] = buttons['rightJS'][1]      # RIGHT: axis-y for forward-and-backward
            base_action[1] = -buttons['rightJS'][0]     # RIGHT: axis-x for move-left-and-right -> non-effective
        if 'leftJS' in buttons:
            base_action[2] = -buttons['leftJS'][0]      # LEFT: axis-x for trun-left-and-right
        action.base = base_action

        # torso command
        if 'leftJS' in buttons:
            action.torso = 0.01 * buttons['leftJS'][1]  # LEFT: axis-y for up-and-down

        return action