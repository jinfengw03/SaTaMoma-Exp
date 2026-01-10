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
        
        # Detected spheres (obstacles / objects)
        self.detected_spheres = None
        self.sphere_timestamp = None
        
        # End-effector tracking
        self.ee_position = None
        self.previous_min_distance = None
        self.current_min_distance = None
        self.distance_history = deque(maxlen=5)
        
        # Shared Control State
        self.current_patterns = []  # Store latest geometric analysis
        self.suggested_target = None # [x, y, z] target for shared control
        
        # Temporary directory
        self.temp_dir = tempfile.mkdtemp()
        print(f'[IntentPredictor] Initialized. Temp dir: {self.temp_dir}', end='\r\n', flush=True)
        print(f'[IntentPredictor] Analysis interval: {analysis_interval}s', end='\r\n', flush=True)
        
        # Check Ollama
        if not self.check_ollama():
            print('[IntentPredictor] WARNING: Ollama not available or model not found.', end='\r\n', flush=True)
        
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

    def update_state(self, image, joints, base_vel=None, torso=None, speech=None, obstacles=None, ee_pose=None):
        """
        Update the current state of the robot for prediction.
        :param image: cv2 image (BGR)
        :param joints: list or array of joint positions
        :param base_vel: list or array of base velocity [vx, vy, wz]
        :param torso: float, torso height
        :param speech: string, optional speech input
        :param obstacles: list of detected spheres [[x,y,z,r], ...]
        :param ee_pose: list/array of End Effector pose [x,y,z,qx,qy,qz,qw]
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
            print(f'[IntentPredictor] Received speech: "{speech}"', end='\r\n', flush=True)
            
        if obstacles is not None:
            # Convert simple list of lists to dict format used by analysis
            self.detected_spheres = []
            for obs in obstacles:
                if len(obs) >= 4:
                    self.detected_spheres.append({
                        'x': obs[0], 'y': obs[1], 'z': obs[2], 'r': obs[3]
                    })
            self.sphere_timestamp = time.time()
            
        if ee_pose is not None:
            # Extract position (x, y, z)
            self.ee_position = np.array(ee_pose[:3])
            
        # Update distances if we have new data
        self.update_ee_target_distance()

    def update_ee_target_distance(self):
        """Update the distance from EE to detected objects"""
        if self.ee_position is None or not self.detected_spheres:
            return

        min_distance = float('inf')
        
        # Assume a generic radius for the end-effector (e.g. gripper size)
        ee_radius = 0.1
        
        for target in self.detected_spheres:
            target_pos = np.array([target['x'], target['y'], target['z']])
            target_r = target['r']
            
            # Center distance
            dist = np.linalg.norm(self.ee_position - target_pos)
            # Surface distance
            surface_dist = dist - ee_radius - target_r
            
            if surface_dist < min_distance:
                min_distance = surface_dist
        
        if min_distance != float('inf'):
            self.previous_min_distance = self.current_min_distance
            self.current_min_distance = min_distance
            
            if self.previous_min_distance is not None:
                change = self.current_min_distance - self.previous_min_distance
                self.distance_history.append(change)

    def get_distance_trend_score(self):
        """
        Calculate score (0-1) based on distance trend.
        Approaching = High score, Retreating = Low score.
        """
        if len(self.distance_history) < 2:
            return 0.5
        
        avg_change = np.mean(list(self.distance_history))
        # Sigmoid mapping: -0.1m change -> ~0.88 score, +0.1m -> ~0.11 score
        score = 1.0 / (1.0 + np.exp(avg_change * 20)) 
        return score

    def analyze_sphere_patterns(self, spheres):
        """Analyze spatial patterns of spheres to guess object types"""
        if not spheres:
            return []
            
        patterns = []
        
        # Sort by distance from End-Effector if available, otherwise origin
        if self.ee_position is not None:
            ref = self.ee_position
            # Prioritize objects close to the hand
            nearest = sorted(spheres, key=lambda s: np.sqrt((s['x']-ref[0])**2 + (s['y']-ref[1])**2 + (s['z']-ref[2])**2))[:5]
        else:
            # Sort by distance from origin (robot base approximation)
            nearest = sorted(spheres, key=lambda s: np.sqrt(s['x']**2 + s['y']**2 + s['z']**2))[:5]
        
        # 1. Vertical Stack
        for i, base in enumerate(nearest):
            group = [base]
            for j, cand in enumerate(nearest):
                if i == j: continue
                dx = abs(cand['x'] - base['x'])
                dy = abs(cand['y'] - base['y'])
                dz = cand['z'] - base['z']
                if dx < 0.1 and dy < 0.1 and 0 < dz < 0.3:
                    group.append(cand)
            
            if len(group) >= 2:
                avg_pos = np.mean([[s['x'], s['y'], s['z']] for s in group], axis=0)
                patterns.append({
                    'type': 'vertical_stack',
                    'description': f'{len(group)} spheres stacked vertically',
                    'possible_objects': ['bottle', 'can', 'snake tube'],
                    'position': avg_pos
                })
                break
        
        # 2. Single Sphere (if no stack found)
        if not patterns and nearest:
            s = nearest[0]
            patterns.append({
                'type': 'single_sphere',
                'description': 'Single isolated object',
                'possible_objects': ['apple', 'ball', 'small item'],
                'position': [s['x'], s['y'], s['z']]
            })
            
        return patterns

    def calculate_weighted_confidence(self, vlm_response):
        """Calculate weighted confidence score"""
        # 1. Context & Speech (70%)
        context_score = 0.5
        resp_lower = vlm_response.lower()
        if 'high confidence' in resp_lower or 'very confident' in resp_lower:
            context_score = 0.9
        elif 'low confidence' in resp_lower or 'uncertain' in resp_lower:
            context_score = 0.3
            
        if self.latest_speech and self.speech_timestamp:
            if time.time() - self.speech_timestamp < self.speech_timeout:
                context_score = min(1.0, context_score + 0.2)
                
        # 2. Object Distance (20%)
        dist_score = 0.5
        if self.current_min_distance is not None:
            # Closer = Higher confidence
            dist_score = np.exp(-self.current_min_distance * 2.0)
            dist_score = max(0.1, min(1.0, dist_score))
            
        # 3. Movement Trend (10%)
        trend_score = self.get_distance_trend_score()
        
        weights = {'context': 0.7, 'distance': 0.2, 'trend': 0.1}
        total = (context_score * 0.7) + (dist_score * 0.2) + (trend_score * 0.1)
        
        return {
            'total': total,
            'breakdown': {'context': context_score, 'distance': dist_score, 'trend': trend_score},
            'weights': weights
        }

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
            
        # --- NEW: Sphere Patterns ---
        if self.detected_spheres:
            self.current_patterns = self.analyze_sphere_patterns(self.detected_spheres)
            if self.current_patterns:
                prompt_parts.append("\n=== Detected 3D Object Structure ===")
                for p in self.current_patterns:
                    pos = p['position']
                    prompt_parts.append(f"- Pattern: {p['type'].upper()} ({p['description']}) at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                    prompt_parts.append(f"  Likely: {', '.join(p['possible_objects'])}")
        
        # --- NEW: Distance Trend ---
        if self.current_min_distance is not None:
            prompt_parts.append(f"\nEnd-Effector to Object Distance: {self.current_min_distance*100:.1f}cm")
            if self.previous_min_distance is not None:
                change = self.current_min_distance - self.previous_min_distance
                trend = "APPROACHING" if change < -0.01 else "RETREATING" if change > 0.01 else "STABLE"
                prompt_parts.append(f"Trend: {trend} ({change*100:.1f}cm change)")
        
        prompt_parts.extend([
            "\nPlease provide:",
            "1. Identify key objects (using the 3D structure info)",
            "2. Predicted operator intent",
            "3. Confidence level (High/Medium/Low)",
            "4. Suggested action (e.g., Approach Table, Pick Chips)",
        ])
        
        return "\n".join(prompt_parts)

    def execute_action(self, action_type):
        # NOTE: In a real integrated system, you might want to call TiagoClient methods here
        # instead of subprocess. But for now, we keep the script calling logic.
        if action_type == "APPROACH_TABLE":
            print("[IntentPredictor] Executing: Approach Table", end='\r\n', flush=True)
            # Placeholder for actual execution logic
            # subprocess.run(["rosrun", "tiago_safety", "approach_table.py"], check=True)
            self.last_executed_action = "APPROACH_TABLE"
            self.last_action_time = time.time()
            
        elif action_type == "PICK_CHIPS":
            print("[IntentPredictor] Executing: Pick Chips", end='\r\n', flush=True)
            # Placeholder for actual execution logic
            self.last_executed_action = "PICK_CHIPS"
            self.last_action_time = time.time()

    def check_and_prompt_action(self, vlm_response, confidence_report=None):
        response_lower = vlm_response.lower()
        action_candidate = None
        action_desc = ""
        
        # Reset suggested target
        self.suggested_target = None
        
        # Use calculated confidence if available, else standard check
        confidence_high = False
        confidence_score = 0.0
        if confidence_report:
            confidence_score = confidence_report['total']
            confidence_high = confidence_score > 0.6
        else:
            confidence_high = ('high confidence' in response_lower or 'very confident' in response_lower)
            confidence_score = 0.9 if confidence_high else 0.5
        
        if not confidence_high:
            return

        target_pattern_type = None
        
        if "table" in response_lower and ("approach" in response_lower or "move" in response_lower):
            action_candidate = "APPROACH_TABLE"
            action_desc = "Move to Table"
            # For table approach, we might look for the centroid of all objects or a large surface (not implemented here)
        
        elif ("chip" in response_lower or "snack" in response_lower) and ("pick" in response_lower or "grasp" in response_lower):
            action_candidate = "PICK_CHIPS"
            action_desc = "Pick Chips"
            target_pattern_type = ['vertical_stack', 'single_sphere'] # Chips might be a can (stack) or bag (sphere blob)
            
        elif ("bottle" in response_lower or "pour" in response_lower):
            action_candidate = "POUR_BOTTLE"
            action_desc = "Pour Bottle"
            target_pattern_type = ['vertical_stack']

        # Determine spatial target based on intent
        if target_pattern_type and self.current_patterns:
            for pat in self.current_patterns:
                if pat['type'] in target_pattern_type:
                    self.suggested_target = pat['position']
                    print(f"[IntentPredictor] >> Locked on Target: {pat['type']} at {self.suggested_target}", end='\r\n', flush=True)
                    break
        
        # If we have a spatial target, we can enable shared control
        if self.suggested_target is not None and confidence_score > 0.75:
             print(f"[IntentPredictor] >> SHARED CONTROL ACTIVE: Assisting towards {action_desc}", end='\r\n', flush=True)
             # We rely on the main loop to read self.suggested_target

        if action_candidate:
            if self.last_executed_action == action_candidate:
                 if self.last_action_time and (time.time() - self.last_action_time) < 60.0:
                     return

            print("\r\n" + "!"*60, end='\r\n', flush=True)
            print(f">>> PROPOSED ACTION: {action_desc}", end='\r\n', flush=True)
            if confidence_report:
                print(f">>> Confidence: {confidence_report['total']*100:.1f}%", end='\r\n', flush=True)
            print(">>> Do you want to execute this action? (y/n): ", end='\r\n', flush=True)
            
            # Non-blocking input is hard in a loop, so we might just print the suggestion
            # or use a separate thread for input if strictly necessary.
            # For safety in a real-time loop, we usually don't block on input().
            print("\r\n(Auto-execution disabled for safety in integrated mode)", end='\r\n', flush=True)
            print("!"*60 + "\r\n", end='\r\n', flush=True)

    def analysis_loop(self):
        print('[IntentPredictor] Analysis thread started', end='\r\n', flush=True)
        while self.running:
            current_time = time.time()
            if (current_time - self.last_analysis_time) >= self.analysis_interval:
                if self.latest_image is not None:
                    try:
                        self.analyze_current_state()
                    except Exception as e:
                        print(f"[IntentPredictor] Analysis Error: {e}", end='\r\n', flush=True)
                    self.last_analysis_time = current_time
                else:
                    print('[IntentPredictor] Waiting for image data...', end='\r\n', flush=True)
            time.sleep(0.1)

    def analyze_current_state(self):
        print('[IntentPredictor] Analyzing current state...', end='\r\n', flush=True)
        if self.latest_image is None: 
            return
            
        image_path = self.save_frame_to_file(self.latest_image, f'frame_{time.time()}.jpg')
        prompt = self.generate_context_prompt()
        response = self.query_llava(image_path, prompt)
        
        try:
            os.remove(image_path)
        except:
            pass
            
        # Calculate confidence
        conf_report = self.calculate_weighted_confidence(response)
        
        # Replace \n with \r\n in response for proper terminal display in raw mode
        response_formatted = response.replace('\n', '\r\n')
        
        print(f'[IntentPredictor] Result:\r\n{response_formatted}', end='\r\n', flush=True)
        print(f'Confidence score: {conf_report["total"]:.2f}', end='\r\n', flush=True)
        print("-" * 40, end='\r\n', flush=True)
        
        self.check_and_prompt_action(response, conf_report)

    def stop(self):
        self.running = False
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
