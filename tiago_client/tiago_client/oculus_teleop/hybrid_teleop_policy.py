# -*- coding: utf-8 -*-

import sys
import tty
import termios
import rospy
import numpy as np
from threading import Thread, Lock

class HybridTeleopPolicy:
    """
    Adapts the HybridTeleop (keyboard control) to the interface expected by TiagoClient.
    Provides get_action(obs) method.
    """
    def __init__(self):
        # State
        # Default to Cartesian to match TiagoClient's IK pipeline
        self.mode = 'CARTESIAN' # 'JOINT' or 'CARTESIAN'
        self.lock = Lock()
        
        # Control parameters
        self.cartesian_step = 0.02
        self.orientation_step = 0.05
        self.joint_step = 0.05
        
        # Current accumulated actions
        # Base uses 3D twist matching TeleopAction: [vx, vy, wz]
        self.base_cmd = [0.0, 0.0, 0.0]
        self.torso_cmd = 0.0        # absolute position, needs sync with obs
        self.gripper_cmd = 0.0      # 0.0 (closed) to 1.0 (open)
        self._last_gripper_cmd = 0.0
        self.head_cmd = [0.0, 0.0]  # [pan, tilt] - absolute positions
        self.head_initialized = False
        
        # For cartesian/joint delta accumulation
        self.cartesian_delta = np.zeros(6) # x, y, z, roll, pitch, yaw
        self.joint_delta = np.zeros(7)
        
        # Initial state flags
        self.torso_initialized = False

        # Keyboard reading setup
        self.settings = termios.tcgetattr(sys.stdin)
        self.running = True
        
        # Start keyboard listener thread
        self.key_thread = Thread(target=self._keyboard_loop)
        self.key_thread.daemon = True
        
    def start(self):
        self.key_thread.start()
        # Silent mode: no console output

    def _print_usage(self):
        # Silent mode: no console output
        pass

    def _get_key(self):
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
            # Handle escape sequences for arrow keys
            if key == '\x1b':  # ESC sequence
                key += sys.stdin.read(2)  # Read the next 2 chars
            return key
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    def _keyboard_loop(self):
        while self.running:
            try:
                key = self._get_key()
                with self.lock:
                    if key == '\x03': # Ctrl+C
                        self.running = False
                        break
                    
                    if key == '\t':
                        self.mode = 'CARTESIAN' if self.mode == 'JOINT' else 'JOINT'
                        self._print_usage()
                        continue

                    # Base (vx, vy, wz) — match VR interface shape to avoid server errors
                    if key == 'w': self.base_cmd = [1.0, 0.0, 0.0]
                    elif key == 's': self.base_cmd = [-1.0, 0.0, 0.0]
                    elif key == 'a': self.base_cmd = [0.0, 0.6, 0.0]
                    elif key == 'd': self.base_cmd = [0.0, -0.6, 0.0]
                    elif key == 'q': self.base_cmd = [0.0, 0.0, 2.5]
                    elif key == 'e': self.base_cmd = [0.0, 0.0, -2.5]
                    elif key == ' ': self.base_cmd = [0.0, 0.0, 0.0]

                    # Torso (delta accumulation request)
                    if key == 'm': self.torso_cmd += 0.02
                    elif key == 'n': self.torso_cmd -= 0.02

                    # Gripper
                    if key == 'p': self.gripper_cmd = 1.0
                    elif key == ';': self.gripper_cmd = 0.0

                    # Head Control (Arrow Keys)
                    # Pan: left/right (head_1_joint), Tilt: up/down (head_2_joint)
                    head_step = 0.1
                    if key == '\x1b[C':  # Right arrow
                        self.head_cmd[0] = max(self.head_cmd[0] - head_step, -1.3)
                    elif key == '\x1b[D':  # Left arrow
                        self.head_cmd[0] = min(self.head_cmd[0] + head_step, 1.3)
                    elif key == '\x1b[A':  # Up arrow
                        self.head_cmd[1] = max(self.head_cmd[1] - head_step, -1.05)
                    elif key == '\x1b[B':  # Down arrow
                        self.head_cmd[1] = min(self.head_cmd[1] + head_step, 0.785)

                    # Arm Control
                    if self.mode == 'CARTESIAN':
                        if key == 'i': self.cartesian_delta[0] += self.cartesian_step
                        elif key == 'k': self.cartesian_delta[0] -= self.cartesian_step
                        elif key == 'j': self.cartesian_delta[1] += self.cartesian_step
                        elif key == 'l': self.cartesian_delta[1] -= self.cartesian_step
                        elif key == 'u': self.cartesian_delta[2] += self.cartesian_step
                        elif key == 'o': self.cartesian_delta[2] -= self.cartesian_step
                        elif key == 'r': self.cartesian_delta[3] += self.orientation_step
                        elif key == 'f': self.cartesian_delta[3] -= self.orientation_step
                        elif key == 't': self.cartesian_delta[4] += self.orientation_step
                        elif key == 'g': self.cartesian_delta[4] -= self.orientation_step
                        elif key == 'y': self.cartesian_delta[5] += self.orientation_step
                        elif key == 'h': self.cartesian_delta[5] -= self.orientation_step
                    elif self.mode == 'JOINT':
                        if key == 'r': self.joint_delta[0] += self.joint_step
                        elif key == 'f': self.joint_delta[0] -= self.joint_step
                        elif key == 't': self.joint_delta[1] += self.joint_step
                        elif key == 'g': self.joint_delta[1] -= self.joint_step
                        elif key == 'y': self.joint_delta[2] += self.joint_step
                        elif key == 'h': self.joint_delta[2] -= self.joint_step
                        elif key == 'u': self.joint_delta[3] += self.joint_step
                        elif key == 'j': self.joint_delta[3] -= self.joint_step
                        elif key == 'i': self.joint_delta[4] += self.joint_step
                        elif key == 'k': self.joint_delta[4] -= self.joint_step
                        elif key == 'o': self.joint_delta[5] += self.joint_step
                        elif key == 'l': self.joint_delta[5] -= self.joint_step
                        elif key == 'z': self.joint_delta[6] += self.joint_step
                        elif key == 'x': self.joint_delta[6] -= self.joint_step

            except Exception as e:
                print(f"Keyboard loop error: {e}")

    def get_action(self, obs, is_filter=False):
        """
        Equivalent interface to TeleopPolicy.get_action().
        Returns a dict-like object (AttrDict) with action components.
        """
        # Initialize torso from observation once
        if not self.torso_initialized and obs.torso is not None:
            self.torso_cmd = obs.torso
            self.torso_initialized = True
        
        # Initialize head from observation once
        if not self.head_initialized and hasattr(obs, 'head') and obs.head is not None:
            self.head_cmd = list(obs.head) if hasattr(obs.head, '__iter__') else [0.0, 0.0]
            self.head_initialized = True

        action = {}
        extra = {'buttons': {}} # Placeholder

        with self.lock:
            # Base
            action['base'] = np.array(self.base_cmd)
            # Reset base after each read so keypress is a pulse, not a latch
            self.base_cmd = [0.0, 0.0, 0.0]
            
            # Torso
            # Clip accumulated torso command
            self.torso_cmd = max(0.0, min(0.35, self.torso_cmd))
            action['torso'] = np.array([self.torso_cmd])

            # Head
            action['head'] = np.array(self.head_cmd)

            # Right Arm
            # We construct a 7+1 vector: 7 joints/pose + 1 gripper
            # Note: TiagoClient expects Cartesian delta [x,y,z, r,p,y] + gripper
            # OR logic in TiagoClient needs to be adapted if we want to send joint deltas directly.
            # However, looking at TiagoClient.get_teleop_action logic:
            # It expects raw_action[side] to be [pos_delta(3), euler_delta(3), gripper(1)]
            
            # Since HybridTeleop supported Joint Mode, but TiagoClient implementation shown 
            # hardcodes "Cartesian -> IK -> Safety" flow, we have two options:
            # 1. Only support Cartesian mode here.
            # 2. Modify TiagoClient to accept Joint commands directly.
            
            # For now, let's implement Cartesian part compatible with existing TiagoClient
            # If in JOINT mode, we might need a workaround or just support Cartesian for now.
            
            if self.mode == 'CARTESIAN':
                # [dx, dy, dz, droll, dpitch, dyaw]
                pose_delta = self.cartesian_delta.copy()
                # Deadband small noise to prevent drift when idle
                if np.all(np.abs(pose_delta) < 1e-6):
                    pose_delta[:] = 0
                # Reset delta after reading (impulse control)
                self.cartesian_delta[:] = 0 
                send_arm = not np.allclose(pose_delta, 0)
                send_arm = send_arm or (self.gripper_cmd != self._last_gripper_cmd)
                if send_arm:
                    cmd_vec = np.concatenate([pose_delta, [self.gripper_cmd]])
                    action['right'] = cmd_vec
                    self._last_gripper_cmd = self.gripper_cmd
                else:
                    action['right'] = None
            else:
                # Fallback or Todo: Handle JOINT mode in TiagoClient
                # Current TiagoClient logic:
                # cartesian_delta = raw_action[side][:6]...
                # So we can't easily pass joint angles without modifying client.
                # We will just print warning for now.
                print("Warning: JOINT mode not fully supported by standard TiagoClient teleop logic yet.")
                action['right'] = None
                # Clear joint delta so it never latches
                self.joint_delta[:] = 0

            # Left Arm (No control in this keyboard map)
            action['left'] = None

        # Wrap in an object that allows access via .extra if needed (simulating TeleopAction)
        class ActionWrapper(dict):
            def __init__(self, *args, **kwargs):
                super(ActionWrapper, self).__init__(*args, **kwargs)
                self.extra = extra

        return ActionWrapper(action)
