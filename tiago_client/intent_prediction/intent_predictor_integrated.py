#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import time
import cv2
import numpy as np
import base64
import tempfile
import os
from collections import deque
from openai import OpenAI  # Added OpenAI SDK

from pathlib import Path
from dotenv import load_dotenv

# 1. Automatically find the path to the .env file
env_path = Path(__file__).resolve().parents[1] / ".env"

# 2. Load the specific .env file
load_dotenv(dotenv_path=env_path)

class IntentPredictorIntegrated:
    def __init__(self, model_name='gpt-4o', analysis_interval=5.0):
        # Retrieve the key
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Security: Print only a small portion of the key to verify it loaded
        if api_key:
            print(f"[SUCCESS] OpenAI API Key loaded. (Starts with: {api_key[:8]}...)")
        else:
            print(f"[CRITICAL ERROR] Failed to load OPENAI_API_KEY from {env_path}")
            # You may want to sys.exit(1) here for safety in a real robot
            
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
        
        # Temporary directory
        self.temp_dir = tempfile.mkdtemp()
        print(f'[IntentPredictor] Initialized with {model_name}. Temp dir: {self.temp_dir}', end='\r\n', flush=True)

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
                                    "detail": "low" # Use 'low' to save tokens/latency on robot
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

    def calculate_weighted_confidence(self, vlm_response):
        context_score = 0.5
        resp_lower = vlm_response.lower()
        if any(x in resp_lower for x in ['high confidence', 'very confident']): context_score = 0.9
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
        prompt_parts = ["Analyze the robot teleoperation scene.", "Describe objects and predict intent based on visual and 3D data."]
    
        # Get the physical movement status
        movement_status = self.get_movement_status()
    
        # Inject it into the prompt context
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
            
        # Ask the VLM to confirm this in the output
        prompt_parts.extend(["You are a mobile manipulator in an indoor household setting. Provide each of the following on new line and format outptut: 1. Key objects (likely indoor household objects),\n 2. Intent,\n 3. Movement Status (Confirm if gripper is approaching), \n 4. Confidence (High/Med/Low)."])
    
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
        image_path = self.save_frame_to_file(self.latest_image, f'frame_{time.time()}.jpg')
        prompt = self.generate_context_prompt()
        # Call the new GPT-4o function
        response = self.query_gpt_vision(image_path, prompt)
        
        try: os.remove(image_path)
        except: pass
            
        conf_report = self.calculate_weighted_confidence(response)
        print(f'[GPT-4o Result]:\n{response}\nConfidence: {conf_report["total"]:.2f}')
        self.check_and_prompt_action(response, conf_report)

    def check_and_prompt_action(self, response, confidence_report):
        pass

    def stop(self):
        self.running = False
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


    def get_movement_status(self):
        """Determines if gripper is moving towards or away from the nearest object."""
        if len(self.distance_history) < 3:
            return "UNKNOWN"
    
        # Calculate average change in distance over the last few frames
        avg_change = np.mean(list(self.distance_history))
    
        # Threshold: -0.001 m/s (1mm) implies movement towards
        if avg_change < -0.002: 
            return "APPROACHING"
        elif avg_change > 0.002:
            return "RETREATING"
        else:
            return "STATIC"