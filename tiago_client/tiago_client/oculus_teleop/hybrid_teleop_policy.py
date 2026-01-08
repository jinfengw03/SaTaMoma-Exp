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
        self.head_step = 0.1
        
        # Current accumulated actions
        # Base uses 3D twist matching TeleopAction: [vx, vy, wz]
        self.base_cmd = [0.0, 0.0, 0.0]
        self.torso_cmd = 0.0        # absolute position, needs sync with obs
        self.gripper_cmd = 0.0      # 0.0 (closed) to 1.0 (open)
        self._last_gripper_cmd = 0.0
        self.head_cmd = np.zeros(2)
        
        # For cartesian/joint delta accumulation
        self.cartesian_delta = np.zeros(6) # x, y, z, roll, pitch, yaw
        self.joint_delta = np.zeros(7)
        
        # Initial state flags
        self.torso_initialized = False
        self.head_initialized = False

        # Head joint limits (rad): [head_1_joint, head_2_joint]
        self.head_limits = [(-1.3, 1.3), (-1.05, 0.785)]

        # Keyboard reading setup
        self.settings = termios.tcgetattr(sys.stdin)
        self.running = True
        
        # Start keyboard listener thread
        self.key_thread = Thread(target=self._keyboard_loop)
        self.key_thread.daemon = True
        
    def start(self):
        self.key_thread.start()
        print("Hybrid Teleop Policy Started (Keyboard)")
        self._print_usage()

    def _print_usage(self):
        # Short, multi-line block to reduce wrap/indent artifacts when mixed with other logs
        if self.mode == 'CARTESIAN':
            arm_help = "Arm: I/K X, J/L Y, U/O Z, R/F Roll, T/G Pitch, Y/H Yaw"
        else:
            arm_help = "Arm: JOINT unsupported"
        lines = [
            f"[KB Teleop] Mode: {self.mode} (TAB to toggle)",
            "Base: WASD move, QE rotate, Space stop | Torso: M up / N down | Head: Arrow keys | Gripper: P open / ; close",
            arm_help,
        ]
        print("\n".join(lines), flush=True)

    def _get_key(self):
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
            if key == '\x1b':
                key += sys.stdin.read(2) # arrow keys
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
                    if key == 'w': self.base_cmd = [0.5, 0.0, 0.0]
                    elif key == 's': self.base_cmd = [-0.5, 0.0, 0.0]
                    elif key == 'a': self.base_cmd = [0.0, 0.3, 0.0]
                    elif key == 'd': self.base_cmd = [0.0, -0.3, 0.0]
                    elif key == 'q': self.base_cmd = [0.0, 0.0, 1.5]
                    elif key == 'e': self.base_cmd = [0.0, 0.0, -1.5]
                    elif key == ' ': self.base_cmd = [0.0, 0.0, 0.0]

                    # Torso (delta accumulation request)
                    if key == 'm': self.torso_cmd += 0.02
                    elif key == 'n': self.torso_cmd -= 0.02

                    # Gripper
                    if key == 'p': self.gripper_cmd = 1.0
                    elif key == ';': self.gripper_cmd = 0.0

                    # Head (absolute position commands)
                    if key == '\x1b[A': # Up
                        self.head_cmd[1] = max(self.head_limits[1][0], self.head_cmd[1] - self.head_step)
                    elif key == '\x1b[B': # Down
                        self.head_cmd[1] = min(self.head_limits[1][1], self.head_cmd[1] + self.head_step)
                    elif key == '\x1b[C': # Right
                        self.head_cmd[0] = max(self.head_limits[0][0], self.head_cmd[0] - self.head_step)
                    elif key == '\x1b[D': # Left
                        self.head_cmd[0] = min(self.head_limits[0][1], self.head_cmd[0] + self.head_step)

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
        if not self.head_initialized and getattr(obs, 'head', None) is not None:
            self.head_cmd = np.array(obs.head).reshape(-1)[:2]
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

            # Head (absolute positions for head_1_joint, head_2_joint)
            if self.head_initialized:
                self.head_cmd[0] = np.clip(self.head_cmd[0], *self.head_limits[0])
                self.head_cmd[1] = np.clip(self.head_cmd[1], *self.head_limits[1])
                action['head'] = self.head_cmd.copy()
            else:
                action['head'] = None

        # Wrap in an object that allows access via .extra if needed (simulating TeleopAction)
        class ActionWrapper(dict):
            def __init__(self, *args, **kwargs):
                super(ActionWrapper, self).__init__(*args, **kwargs)
                self.extra = extra

        return ActionWrapper(action)
