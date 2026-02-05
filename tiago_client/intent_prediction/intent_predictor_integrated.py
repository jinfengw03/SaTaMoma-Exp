#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import time
import cv2
import numpy as np
import base64
import tempfile
import os
import re
import shutil
from collections import deque
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# 1. Automatically find the path to the .env file
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

class IntentPredictorIntegrated:
    def __init__(self, model_name='gpt-4o', analysis_interval=5.0):
        # Time tracking for the entire session
        self.task_start_time = time.time()
        
        # Retrieve the key
        api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            print(f"[SUCCESS] OpenAI API Key loaded. (Starts with: {api_key[:8]}...)")
        else:
            print(f"[CRITICAL ERROR] Failed to load OPENAI_API_KEY from {env_path}")
            
        self.client = OpenAI(api_key=api_key)
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
        
        # Speech input
        self.latest_speech = None
        self.speech_timestamp = None
        self.speech_timeout = 30.0
        
        # Detected spheres
        self.detected_spheres = None
        self.sphere_timestamp = None
        
        # End-effector tracking
        self.ee_position = None
        self.previous_min_distance = None
        self.current_min_distance = None
        self.distance_history = deque(maxlen=5)
        
        # Shared Control State
        self.current_patterns = []
        self.suggested_target = None 
        self.last_vlm_response = ""
        self.current_confidence = {'total': 0.5}
        
        # Temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Log file initialization with Task Start Timestamp
        self.log_file = os.path.join(os.getcwd(), f"vlm_response_log_{int(self.task_start_time)}.txt")
        print(f"[IntentPredictor] Initialized. Temp dir: {self.temp_dir}")
        print(f"[IntentPredictor] Task Started at: {time.ctime(self.task_start_time)}")
        print(f"[IntentPredictor] Logging to: {self.log_file}")
        
        with open(self.log_file, "w") as f:
            f.write(f"=== ROBOT TASK LOG ===\n")
            f.write(f"TASK START TIME: {time.ctime(self.task_start_time)}\n")
            f.write(f"MODEL USED: {self.model_name}\n")
            f.write(f"{'='*40}\n\n")

        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self.analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def encode_image(self, image_path):
        """Helper to convert image file to base64 for GPT-4o"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def query_gpt_vision(self, image_path, prompt):
        """Replaces query_llava with OpenAI API calls"""
        try:
            base64_image = self.encode_image(image_path)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                }
                            },
                        ],
                    }
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to OpenAI: {str(e)}"

    def update_state(self, image, joints, base_vel=None, torso=None, speech=None, obstacles=None, ee_pose=None):
        if image is not None:
            self.latest_image = image
        if joints is not None:
            self.current_joint_positions = np.array(joints)
            state_entry = {'time': time.time(), 'positions': self.current_joint_positions}
            if base_vel is not None: state_entry['base_vel'] = np.array(base_vel)
            if torso is not None: state_entry['torso'] = torso
            self.joint_history.append(state_entry)
        if speech is not None:
            self.latest_speech = speech
            self.speech_timestamp = time.time()
        if obstacles is not None:
            self.detected_spheres = [{'x': o[0], 'y': o[1], 'z': o[2], 'r': o[3]} for o in obstacles if len(o) >= 4]
            self.sphere_timestamp = time.time()
        if ee_pose is not None:
            self.ee_position = np.array(ee_pose[:3])
        self.update_ee_target_distance()
        self.current_confidence = self.calculate_weighted_confidence()

    def update_ee_target_distance(self):
        if self.ee_position is None or not self.detected_spheres: return
        min_distance = float('inf')
        ee_radius = 0.1
        for target in self.detected_spheres:
            target_pos = np.array([target['x'], target['y'], target['z']])
            dist = np.linalg.norm(self.ee_position - target_pos)
            surface_dist = dist - ee_radius - target['r']
            if surface_dist < min_distance: min_distance = surface_dist
        if min_distance != float('inf'):
            self.previous_min_distance = self.current_min_distance
            self.current_min_distance = min_distance
            if self.previous_min_distance is not None:
                self.distance_history.append(self.current_min_distance - self.previous_min_distance)

    def get_distance_trend_score(self):
        if len(self.distance_history) < 2: return 0.5
        avg_change = np.mean(list(self.distance_history))
        return 1.0 / (1.0 + np.exp(avg_change * 20))

    def analyze_sphere_patterns(self, spheres):
        if not spheres: return []
        patterns = []
        ref = self.ee_position if self.ee_position is not None else np.zeros(3)
        nearest = sorted(spheres, key=lambda s: np.linalg.norm(np.array([s['x'], s['y'], s['z']]) - ref))[:5]
        for i, base in enumerate(nearest):
            group = [base]
            for j, cand in enumerate(nearest):
                if i == j: continue
                if abs(cand['x']-base['x']) < 0.1 and abs(cand['y']-base['y']) < 0.1 and 0 < (cand['z']-base['z']) < 0.3:
                    group.append(cand)
            if len(group) >= 2:
                avg_pos = np.mean([[s['x'], s['y'], s['z']] for s in group], axis=0)
                patterns.append({'type': 'vertical_stack', 'description': f'{len(group)} spheres stacked', 'possible_objects': ['bottle', 'can'], 'position': avg_pos})
                break
        if not patterns and nearest:
            patterns.append({'type': 'single_sphere', 'description': 'Single object', 'possible_objects': ['apple', 'ball'], 'position': [nearest[0]['x'], nearest[0]['y'], nearest[0]['z']]})
        return patterns

    def calculate_weighted_confidence(self, vlm_response=None):
        if vlm_response is not None:
            self.last_vlm_response = vlm_response

        context_score = 0.5
        resp_lower = self.last_vlm_response.lower()
        
        match = re.search(r'confidence.*?(\d+(?:\.\d+)?)', resp_lower)
        if match:
             val = float(match.group(1))
             if val > 1.0: val /= 100.0
             context_score = max(0.0, min(1.0, val))
        elif any(x in resp_lower for x in ['high confidence', 'very confident']): context_score = 0.7
        elif any(x in resp_lower for x in ['low confidence', 'uncertain']): context_score = 0.3
        
        if self.latest_speech and self.speech_timestamp and (time.time() - self.speech_timestamp < self.speech_timeout):
            context_score = min(1.0, context_score + 0.2)
        dist_score = np.exp(-self.current_min_distance * 2.0) if self.current_min_distance else 0.5
        trend_score = self.get_distance_trend_score()
        total = (context_score * 0.7) + (max(0.1, min(1.0, dist_score)) * 0.2) + (trend_score * 0.1)
        return {'total': total}

    def save_frame_to_file(self, image, filename):
        filepath = os.path.join(self.temp_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath

    def generate_context_prompt(self):
        prompt_parts = ["Analyze the robot teleoperation scene.", "Describe objects and predict intent based on visual and 3D data.", "Update confidence percentage based on context in each frame."]
        movement_status = self.get_movement_status()
        if movement_status == "APPROACHING":
            prompt_parts.append("TELEMETRY: Gripper is physically MOVING CLOSER to an object.")
        elif movement_status == "RETREATING":
            prompt_parts.append("TELEMETRY: Gripper is MOVING AWAY from an object.")
        if self.last_executed_action == "APPROACH_TABLE" and (time.time() - (self.last_action_time or 0)) < 60.0:
            prompt_parts.append("CONTEXT: Robot recently approached table; likely preparing to PICK.")
        if self.detected_spheres:
            self.current_patterns = self.analyze_sphere_patterns(self.detected_spheres)
            for p in self.current_patterns:
                prompt_parts.append(f"3D Structure: {p['type']} at {p['position']}. Possible: {p['possible_objects']}")
        prompt_parts.extend(["You are a mobile manipulator in an indoor household setting. Provide each of the following on new line and format outptut: 1. Key objects,\n 2. Intent,\n 3. Movement Status,\n 4. Confidence (0.0-1.0)."])
        return "\n".join(prompt_parts)

    def analysis_loop(self):
        while self.running:
            current_time = time.time()
            if (current_time - self.last_analysis_time) >= self.analysis_interval:
                if self.latest_image is not None:
                    try:
                        self.analyze_current_state()
                    except Exception as e:
                        print(f"[IntentPredictor] Error: {e}")
                    self.last_analysis_time = current_time
            time.sleep(0.1)

    def analyze_current_state(self):
        # --- Start Time Tracking for this specific prediction ---
        prediction_start = time.time()
        start_human = time.strftime('%H:%M:%S', time.localtime(prediction_start))
        print(f"\n[VLM] Prediction cycle STARTED at {start_human}")

        image_path = self.save_frame_to_file(self.latest_image, f'frame_{prediction_start}.jpg')
        prompt = self.generate_context_prompt()
        
        # Call VLM
        response = self.query_gpt_vision(image_path, prompt)
        
        # --- End Time Tracking ---
        prediction_end = time.time()
        end_human = time.strftime('%H:%M:%S', time.localtime(prediction_end))
        latency = prediction_end - prediction_start
        
        # Log the response and time stamps
        if hasattr(self, 'log_file'):
            try:
                with open(self.log_file, "a") as f:
                    f.write(f"--- Prediction Cycle ---\n")
                    f.write(f"VLM Prediction Start: {start_human}\n")
                    f.write(f"VLM Prediction End:   {end_human}\n")
                    f.write(f"Cycle Latency:        {latency:.2f} seconds\n")
                    f.write(f"Time Since Task Start: {prediction_end - self.task_start_time:.2f} seconds\n")
                    f.write(f"PROMPT:\n{prompt}\n")
                    f.write(f"RESPONSE:\n{response}\n")
                    f.write("-" * 40 + "\n")
            except Exception as e:
                print(f"Error logging VLM response: {e}")
        
        try: os.remove(image_path)
        except: pass
            
        conf_report = self.calculate_weighted_confidence(response)
        print(f"[VLM] Prediction FINISHED at {end_human} (Took {latency:.2f}s)")
        print(f'[GPT-4o Result]:\n{response}\nConfidence: {conf_report["total"]:.2f}')
        
        self.check_and_prompt_action(response, conf_report)

    def check_and_prompt_action(self, response, confidence_report):
        pass

    def stop(self):
        """Cleanly stops the thread and logs the final task finish time."""
        self.running = False
        finish_time = time.time()
        total_duration = finish_time - self.task_start_time
        
        if hasattr(self, 'log_file'):
            try:
                with open(self.log_file, "a") as f:
                    f.write(f"\n{'='*40}\n")
                    f.write(f"TASK FINISHED AT: {time.ctime(finish_time)}\n")
                    f.write(f"TOTAL SESSION DURATION: {total_duration:.2f} seconds\n")
                    f.write(f"{'='*40}\n")
            except Exception as e:
                print(f"Error writing final log: {e}")

        print(f"[IntentPredictor] Task finished at {time.ctime(finish_time)}. Total: {total_duration:.2f}s")
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def get_movement_status(self):
        if len(self.distance_history) < 3: return "UNKNOWN"
        avg_change = np.mean(list(self.distance_history))
        if avg_change < -0.002: return "APPROACHING"
        elif avg_change > 0.002: return "RETREATING"
        else: return "STATIC"