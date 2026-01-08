#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import time
import cv2
import numpy as np
import subprocess
import tempfile
import os
from collections import deque

class IntentPredictorIntegrated:
    def __init__(self, model_name='llava:7b', analysis_interval=5.0):
        self.model_name = model_name
        self.analysis_interval = analysis_interval
        
        # State variables
        self.latest_image = None
        self.current_joint_positions = None
        self.joint_history = deque(maxlen=30)
        self.last_analysis_time = 0
        self.running = True
        
        # Action history
        self.last_executed_action = None
        self.last_action_time = None
        
        # Speech input (can be updated externally)
        self.latest_speech = None
        self.speech_timestamp = None
        self.speech_timeout = 30.0
        
        # Temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Check Ollama
        if not self.check_ollama():
            print('[IntentPredictor] WARNING: Ollama not available or model not found.')
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self.analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def check_ollama(self):
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
            return True
        except Exception:
            return False

    def update_state(self, image, joints, base_vel=None, torso=None, speech=None):
        """
        Update the current state of the robot for prediction.
        :param image: cv2 image (BGR)
        :param joints: list or array of joint positions
        :param base_vel: list or array of base velocity [vx, vy, wz]
        :param torso: float, torso height
        :param speech: string, optional speech input
        """
        if image is not None:
            self.latest_image = image
        
        if joints is not None:
            self.current_joint_positions = np.array(joints)
            
            state_entry = {
                'time': time.time(),
                'positions': self.current_joint_positions
            }
            if base_vel is not None:
                state_entry['base_vel'] = np.array(base_vel)
            if torso is not None:
                state_entry['torso'] = torso
                
            self.joint_history.append(state_entry)
            
        if speech is not None:
            self.latest_speech = speech
            self.speech_timestamp = time.time()
            print(f'[IntentPredictor] Received speech: "{speech}"')

    def save_frame_to_file(self, image, filename):
        filepath = os.path.join(self.temp_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath

    def query_llava(self, image_path, prompt):
        try:
            full_prompt = f"{prompt}\n{image_path}"
            result = subprocess.run(
                ["ollama", "run", self.model_name, full_prompt],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return f"Error: {result.stderr.strip()}"
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_context_prompt(self):
        prompt_parts = [
            "You are analyzing a robot teleoperation scenario.",
            "Describe what you see in the image and predict the operator's intent."
        ]
        
        # Context: Previous action
        if self.last_executed_action == "APPROACH_TABLE" and self.last_action_time:
             if (time.time() - self.last_action_time) < 60.0:
                 prompt_parts.append("CONTEXT: The robot has just successfully approached the table.")
                 prompt_parts.append("This sequence strongly implies the user now wants to PICK an object (e.g., chips) from the table.")
                 prompt_parts.append("")
        # Context: Motion
        motion_desc = "unknown"
        base_desc = "unknown"
        torso_desc = "unknown"
        
        if len(self.joint_history) >= 5:
            start_state = self.joint_history[0]
            end_state = self.joint_history[-1]
            
            # Arm motion
            max_diff = np.max(np.abs(end_state['positions'] - start_state['positions']))
            if max_diff > 0.02:
                motion_desc = "MOVING"
            else:
                motion_desc = "STATIONARY"
                
            # Base motion
            if 'base_vel' in end_state:
                base_vel = end_state['base_vel']
                # Check linear and angular velocity
                if np.linalg.norm(base_vel[:2]) > 0.05 or abs(base_vel[2]) > 0.1:
                    base_desc = "MOVING"
                else:
                    base_desc = "STATIONARY"
            
            # Torso motion
            if 'torso' in end_state and 'torso' in start_state:
                if abs(end_state['torso'] - start_state['torso']) > 0.01:
                    torso_desc = "MOVING"
                else:
                    torso_desc = "STATIONARY"
        
        prompt_parts.append(f"Robot Arm Status: {motion_desc}")
        if base_desc != "unknown":
            prompt_parts.append(f"Robot Base Status: {base_desc}")
        if torso_desc != "unknown":
            prompt_parts.append(f"Robot Torso Status: {torso_desc}")
        
        
        prompt_parts.append(f"Robot Arm Status: {motion_desc}")
        
        prompt_parts.extend([
            "Please provide:",
            "1. Identify key objects",
            "2. Predicted operator intent",
            "3. Confidence level (High/Medium/Low)",
            "4. Suggested action (e.g., Approach Table, Pick Chips)",
        ])
        
        return "\n".join(prompt_parts)

    def execute_action(self, action_type):
        # NOTE: In a real integrated system, you might want to call TiagoClient methods here
        # instead of subprocess. But for now, we keep the script calling logic.
        if action_type == "APPROACH_TABLE":
            print("[IntentPredictor] Executing: Approach Table")
            # Placeholder for actual execution logic
            # subprocess.run(["rosrun", "tiago_safety", "approach_table.py"], check=True)
            self.last_executed_action = "APPROACH_TABLE"
            self.last_action_time = time.time()
            
        elif action_type == "PICK_CHIPS":
            print("[IntentPredictor] Executing: Pick Chips")
            # Placeholder for actual execution logic
            self.last_executed_action = "PICK_CHIPS"
            self.last_action_time = time.time()

    def check_and_prompt_action(self, vlm_response):
        response_lower = vlm_response.lower()
        action_candidate = None
        action_desc = ""
        confidence_high = ('high confidence' in response_lower or 'very confident' in response_lower)
        
        has_speech = (self.latest_speech is not None and 
                     self.speech_timestamp is not None and 
                     (time.time() - self.speech_timestamp) < self.speech_timeout)

        if not confidence_high and not has_speech:
            return

        if "table" in response_lower and ("approach" in response_lower or "move" in response_lower):
            action_candidate = "APPROACH_TABLE"
            action_desc = "Move to Table"
        elif ("chip" in response_lower or "snack" in response_lower) and ("pick" in response_lower or "grasp" in response_lower):
            action_candidate = "PICK_CHIPS"
            action_desc = "Pick Chips"

        if action_candidate:
            if self.last_executed_action == action_candidate:
                 if self.last_action_time and (time.time() - self.last_action_time) < 60.0:
                     return

            print("\n" + "!"*60)
            print(f">>> PROPOSED ACTION: {action_desc}")
            if has_speech:
                print(f">>> Based on Speech: \"{self.latest_speech}\"")
            print(">>> Do you want to execute this action? (y/n): ", end='', flush=True)
            
            # Non-blocking input is hard in a loop, so we might just print the suggestion
            # or use a separate thread for input if strictly necessary.
            # For safety in a real-time loop, we usually don't block on input().
            print("\n(Auto-execution disabled for safety in integrated mode)")
            print("!"*60 + "\n")

    def analysis_loop(self):
        while self.running:
            current_time = time.time()
            if (current_time - self.last_analysis_time) >= self.analysis_interval:
                if self.latest_image is not None:
                    self.analyze_current_state()
                    self.last_analysis_time = current_time
            time.sleep(0.1)

    def analyze_current_state(self):
        print('[IntentPredictor] Analyzing current state...')
        image_path = self.save_frame_to_file(self.latest_image, f'frame_{time.time()}.jpg')
        prompt = self.generate_context_prompt()
        response = self.query_llava(image_path, prompt)
        
        try:
            os.remove(image_path)
        except:
            pass
            
        print(f'[IntentPredictor] Result:\n{response}\n{"-"*40}')
        self.check_and_prompt_action(response)

    def stop(self):
        self.running = False
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
