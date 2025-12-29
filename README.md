# TIAGo Client & Server
Code for TIAGo client running on TIAGo development PC and TIAGo server running on host PC that runs model inference and etc.
## Usage instruction
To run run_tiago_sim.oy
```
# Deactivate conda environment before next sourcing command
source /opt/ros/noetic/setup.bash
conda activate ros39
python run_tiago_sim.py
```
# Safety

## Terminal 1: Launch Gazebo

## Terminal 2: Spawn spheres in Gazebo
```bash
python tiago_client/tiago_client/tiago_safety/spawn_spheres.py
```

## Terminal 3: Publish spheres' positions and radius to run_tiago_sim.py
```bash
python tiago_client/tiago_client/tiago_safety/more_sphere.py
```

# Intent Prediction

## Modifications

```python
                # 3. Update Intent Predictor with latest observation
                # obs usually contains 'tiago_head_image' if configured in server
                if 'tiago_head_image' in obs:
                    # Assuming image is decoded or needs decoding. 
                    # If it's raw bytes/base64, it might need processing in TiagoClient first.
                    # Here we assume obs['tiago_head_image'] is a numpy array (H,W,3)
                    img = obs['tiago_head_image']
                    
                    # Get joint positions for context
                    joints = obs.get('right_joints', [])
                    
                    # Update predictor state (non-blocking)
                    predictor.update_state(image=img, joints=joints)
```

Call `client.step` in `run_tiago_real.py` (above) to get new observation data:

```python
        obs = decode4json(recept_json['obs'])
        info = decode4json(recept_json['info'])
        return obs, info
```

?Here we assume obs['tiago_head_image'] is a numpy array (H,W,3)? Try first

## Speech inputs haven't been added

## No autonomous execution

## Run
### 1. Download Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
### 2. Pull 7b

```bash
ollama pull llava:7b
```
### 3. Test in real

```bash
python tiago_client/run_tiago_real.py
```
