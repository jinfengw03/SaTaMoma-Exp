import os
import requests
import numpy as np
from collections import OrderedDict
from tiago_client.utils.flask_comm import decode4json, encode2json, reconstruct_space_dict
from tiago_client.oculus_teleop.teleop_policy import TeleopPolicy
from tiago_client.oculus_teleop.teleop_core import TeleopObservation
from tiago_client.oculus_teleop.goal_step_assist import GoalStepAssist, GoalStepAssistConfig
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
            # Check if we want VR or Keyboard (Hybrid)
            # You can control this via an env var or argument, for now defaulting to VR if not specified
            teleop_type = os.environ.get("TIAGO_TELEOP_TYPE", "VR") # VR or KEYBOARD
            self.teleop_type = teleop_type
            
            if teleop_type == "KEYBOARD":
                from tiago_client.oculus_teleop.hybrid_teleop_policy import HybridTeleopPolicy
                self.teleop = HybridTeleopPolicy()
                self._teleop_needs_start = True  # Defer start to avoid terminal corruption
            else:
                from tiago_client.oculus_teleop.configs.only_vr import teleop_config
                self.teleop = TeleopPolicy(teleop_config)
                self.teleop.start()
                self._teleop_needs_start = False
            
            # Initialize IK and Safety for both arms
            self.ik_solvers = {
                'right': TiagoIK(side='right'),
                'left': TiagoIK(side='left')
            }
            
            # Path to URDF (assuming it's in the same package)
            urdf_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'urdf')
            urdf_right_arm_path = os.path.join(urdf_dir, 'tiago_right_arm.urdf')
            urdf_left_arm_path = os.path.join(urdf_dir, 'tiago_left_arm.urdf')
            self.safety_filters = {
                'right': JointSafetyFilter(urdf_right_arm_path, side='right'),
                'left': JointSafetyFilter(urdf_left_arm_path, side='left')
            }

            # Optional: "reach-then-step" goal assistance (client-side, ROS-agnostic)
            self.goal_step_assist = GoalStepAssist(GoalStepAssistConfig())
            self._goal_step_manual_active_last = False
            self._goal_step_last_manual_joint_target = None
            self._goal_step_last_manual_gripper = 0.0
        else:
            self._teleop_needs_start = False
            self.teleop_type = None
            self.goal_step_assist = GoalStepAssist(GoalStepAssistConfig())
            self._goal_step_manual_active_last = False
            self._goal_step_last_manual_joint_target = None
            self._goal_step_last_manual_gripper = 0.0
        
        print("[TiagoClient] Connection established and spaces initialized.")

    def start_teleop(self):
        """Start the teleoperation interface after all initialization printing is complete."""
        if hasattr(self, '_teleop_needs_start') and self._teleop_needs_start:
            self.teleop.start()
            self._teleop_needs_start = False

    @staticmethod
    def _summarize_payload(payload):
        summary = {}
        if not isinstance(payload, dict):
            return f"non-dict payload type={type(payload)}"
        for k, v in payload.items():
            try:
                if isinstance(v, np.ndarray):
                    flat = v.reshape(-1)
                    preview = flat[:3].tolist()
                    summary[k] = f"ndarray shape={v.shape} dtype={v.dtype} preview={preview}"
                elif isinstance(v, (list, tuple)):
                    preview = list(v[:3]) if len(v) >= 3 else list(v)
                    summary[k] = f"list len={len(v)} preview={preview}"
                else:
                    summary[k] = f"{type(v).__name__} value={v}"
            except Exception as exc:  # pragma: no cover - defensive
                summary[k] = f"error summarizing: {exc}"
        return summary

    def _post_json(self, endpoint, payload=None, timeout=5.0):
        """POST to the server and decode JSON with helpful errors."""
        url = self.url + endpoint
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except Exception as exc:
            raise RuntimeError(f"[TiagoClient] POST {url} failed: {exc}") from exc
        if not resp.ok:
            raise RuntimeError(
                f"[TiagoClient] POST {url} returned {resp.status_code}: {resp.text[:300]}"
            )
        try:
            return resp.json()
        except Exception as exc:
            snippet = resp.text[:300]
            raise RuntimeError(
                f"[TiagoClient] POST {url} returned non-JSON body: {snippet}"
            ) from exc

    def _obtain_state_space(self):
        data = self._post_json("tiago_get_state_space")
        return reconstruct_space_dict(data)
    
    def _obtain_observation_space(self):
        data = self._post_json("tiago_get_observation_space")
        return reconstruct_space_dict(data)
    
    def _obtain_action_space(self):
        data = self._post_json("tiago_get_action_space")
        return reconstruct_space_dict(data)

    def reset(self, reset_pose):
        """
        Resets the robot to a specified pose.
        :param reset_pose: Dictionary containing target positions for arms, base, torso, etc.
        """
        reset_pose_json = encode2json(reset_pose)
        recept_json = self._post_json("tiago_reset", payload={'reset_pose': reset_pose_json})
        return decode4json(recept_json)

    def step(self, action):
        """
        Sends a control action to the robot.
        :param action: Dictionary of actions (Cartesian or Joint).
        :return: (observation, info)
        """
        action_json = encode2json(action)
        # print("[TiagoClient] -> /tiago_step action summary:", self._summarize_payload(action))
        recept_json = self._post_json(
            "tiago_step",
            payload={'action': action_json},
            timeout=10.0
        )
        # print("[TiagoClient] <- /tiago_step response keys:", list(recept_json.keys()))
        
        obs = decode4json(recept_json['obs'])
        info = decode4json(recept_json['info'])
        return obs, info

    def get_state(self):
        """Retrieves the full state of the robot including visual data."""
        data = self._post_json("tiago_get_state")
        return decode4json(data)
    
    def get_state_wo_vis(self):
        """Retrieves the robot state without heavy visual data (joints, poses only)."""
        data = self._post_json("tiago_get_state_wo_vis")
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

    def get_teleop_action(self, is_filter=False, obstacles=None, assist_target=None, goal_step_target=None, goal_step_enabled=None):
        """
        Reads the current VR controller input and calculates the robot action.
        :param is_filter: Whether to use the teleop policy's internal filter
        :param obstacles: List of [x, y, z, r] for the safety filter
        :param assist_target: Optional [x, y, z] target to attract the hand towards
        :param goal_step_target: Optional [x, y, z] goal point (same frame as EE pose, typically torso_lift_link)
        :param goal_step_enabled: Optional bool to enable/disable "reach-then-step" assistance
        :return: (safe_action, buttons)
        """
        if self.teleop is None:
            return None, {}

        if goal_step_enabled is not None:
            self.goal_step_assist.config.enabled = bool(goal_step_enabled)
        if goal_step_target is not None:
            self.goal_step_assist.set_goal_xyz(goal_step_target)
            
        # Get current robot state
        state = self.get_state_wo_vis()
        
        # Use obstacles from server state if not provided explicitly
        if obstacles is None:
            obstacles = state.get('obstacles')
        
        # Prepare observation for teleop policy
        torso_val = state.get('torso')
        if isinstance(torso_val, np.ndarray):
            torso_val = float(torso_val.reshape(-1)[0]) if torso_val.size else 0.0
        elif isinstance(torso_val, list):
            torso_val = torso_val[0] if torso_val else 0.0
        obs = TeleopObservation(
            left=state.get('left'),
            right=state.get('right'),
            base=state.get('base_pose'),
            torso=torso_val
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
                
                # --- Shared Control / Assistance ---
                if assist_target is not None and side == 'right': # Assuming dominant hand for now
                    # Simple linear blending or attraction
                    # Pull target_pos towards assist_target
                    alpha = 0.05 # Strength of assistance (0-1)
                    target_pos = target_pos + alpha * (np.array(assist_target) - target_pos)
                
                # 2. Local IK
                joints_curr = state.get(f'{side}_joints')
                if joints_curr is not None:
                    joint_goal = self.ik_solvers[side].find_ik(target_pos, target_quat, joints_curr)
                    '''
                    Next time may comment out the safety filter for testing
                    '''
                    if joint_goal is not None:
                        # 3. Safety Filter
                        # if obstacles is not None:
                        #     self.safety_filters[side].update_obstacles(obstacles)
                        
                        # joint_safe = self.safety_filters[side].filter(joints_curr, joint_goal)

                        # 4. Combine with gripper (8 elements total)
                        safe_action[side] = np.concatenate([joint_goal, [gripper_val]])

                        # Track manual control end -> arm a goal step (right arm only)
                        if (
                            side == 'right'
                            and self.goal_step_assist.config.enabled
                            and self.goal_step_assist.goal_xyz is not None
                        ):
                            manual_active = float(np.linalg.norm(np.asarray(cartesian_delta, dtype=float))) > 1e-3
                            if manual_active:
                                self._goal_step_last_manual_joint_target = np.asarray(joint_goal, dtype=float).copy()
                                self._goal_step_last_manual_gripper = float(gripper_val)
                            if (
                                self._goal_step_manual_active_last
                                and (not manual_active)
                                and (self._goal_step_last_manual_joint_target is not None)
                            ):
                                self.goal_step_assist.notify_manual_arm_command(
                                    self._goal_step_last_manual_joint_target,
                                    self._goal_step_last_manual_gripper,
                                )
                            self._goal_step_manual_active_last = manual_active
                    else:
                        # If IK fails, stay at current joints
                        # Use \r\n for proper line breaks when terminal is in raw mode (keyboard teleop)
                        msg = f"[TiagoClient] IK failed for {side} arm. Using current joints."
                        if hasattr(self, '_teleop_needs_start'):
                            # Keyboard mode: need \r\n for proper line breaks
                            print(msg, end='\r\n', flush=True)
                        else:
                            print(msg)
                        safe_action[side] = np.concatenate([joints_curr, [gripper_val]])

                        if (
                            side == 'right'
                            and self.goal_step_assist.config.enabled
                            and self.goal_step_assist.goal_xyz is not None
                        ):
                            manual_active = float(np.linalg.norm(np.asarray(cartesian_delta, dtype=float))) > 1e-3
                            if manual_active:
                                self._goal_step_last_manual_joint_target = np.asarray(joints_curr, dtype=float).copy()
                                self._goal_step_last_manual_gripper = float(gripper_val)
                            if (
                                self._goal_step_manual_active_last
                                and (not manual_active)
                                and (self._goal_step_last_manual_joint_target is not None)
                            ):
                                self.goal_step_assist.notify_manual_arm_command(
                                    self._goal_step_last_manual_joint_target,
                                    self._goal_step_last_manual_gripper,
                                )
                            self._goal_step_manual_active_last = manual_active

        # Optional: inject one extra step toward a fixed goal (right arm)
        if (
            self.goal_step_assist.config.enabled
            and (self.goal_step_assist.goal_xyz is not None)
            and (not self._goal_step_manual_active_last)
        ):
            right_pose = state.get('right')
            right_joints = state.get('right_joints')
            if right_pose is not None and right_joints is not None and len(right_pose) >= 7:
                step = self.goal_step_assist.maybe_compute_goal_step(
                    current_joints=right_joints,
                    current_pos=right_pose[:3],
                    current_quat=right_pose[3:7],
                )
                if step is not None:
                    target_pos, target_quat, gripper_val = step
                    joint_goal = self.ik_solvers['right'].find_ik(target_pos, target_quat, right_joints)
                    if joint_goal is not None:
                        safe_action['right'] = np.concatenate([np.asarray(joint_goal, dtype=float), [gripper_val]])
                    else:
                        safe_action['right'] = np.concatenate([np.asarray(right_joints, dtype=float), [gripper_val]])
        
        # Process base, torso, and head (direct pass-through)
        if 'base' in raw_action:
            safe_action['base'] = raw_action['base']
        if 'torso' in raw_action:
            safe_action['torso'] = raw_action['torso']
        if 'head' in raw_action:
            safe_action['head'] = raw_action['head']
            
        return safe_action, buttons
