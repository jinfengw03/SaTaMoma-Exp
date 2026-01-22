# TeleWithGoal Branch: Assisted Teleoperation with Intent Prediction

This branch implements a **Shared Control** framework for the TIAGo robot, enhancing standard VR teleoperation with AI-driven Intent Prediction and Goal Assistance.

## Key Features

### 1. Integrated Intent Prediction
The system predicts the user's intent by fusing multiple modalities:
- **Vision Language Model (VLM)**: Uses `LLaVA` (via Ollama) to analyze the visual scene and user context.
- **Geometric Analysis**: Detects object patterns from 3D spheres (e.g., "Vertical Stack" implies a bottle, "Single Sphere" implies a ball/apple).
- **Distance Trends**: Tracks if the End-Effector is approaching or retreating from objects.
- **Weighted Confidence**: Combines VLM output, object proximity, and movement history into a single confidence score.

### 2. Shared Control (Assistance)
When confidence is high (>0.75), the system activates **Goal Assistance**:
- **Target Locking**: Automatically identifies the target object (e.g., the specific bottle being reached for).
- **Haptic/Motion Guidance**: Applies a gentle "attraction force" to the user's input, guiding the robot hand towards the target while maintaining user authority.

### 3. Safety Filter Improvements
- Fixed JAX/NumPy type conflicts in the Control Barrier Function (CBF) implementation.
- Ensures collision avoidance with dynamic obstacles (spheres).

## Prerequisites

- **Ollama**: Must be installed and running.
- **Model**: Pull the LLaVA model:
  ```bash
  ollama pull llava:7b
  ```
- **ROS Noetic**: For Robot communication.

## Usage

### 1. Start the Server (Robot Side)
```bash
cd tiago_server
./run_tiago_server.sh
```

### 2. Start the Client (Control Side)
```bash
# Ensure you are in the conda environment (e.g. ros39)
conda activate ros39
python tiago_client/run_tiago_real.py
```

## Optional: "Reach-Then-Step" Goal Assistance

In addition to the existing continuous "attraction" assistance, the client also supports a discrete helper:
after you finish a manual move and the right arm reaches that commanded joint target, it will take **one extra small step** toward a fixed goal point.

- Implemented in `tiago_client/tiago_client/oculus_teleop/goal_step_assist.py`
- Wired into `TiagoClient.get_teleop_action()` and `TiagoClientSim.get_teleop_action()`
- `goal_step_target` is a 3D point expressed in the same frame as the client's EE pose (typically `torso_lift_link` in this repo)

Example usage (call-site side):

```python
safe_action, buttons = client.get_teleop_action(
  goal_step_enabled=True,
  goal_step_target=[0.68, -0.221, 1.06],
)
```

### Using RViz to set the goal (debug-friendly)

You can set `goal_step_target` by clicking a point/goal in RViz, and letting the client TF-transform it into `torso_lift_link`:

- Set env var: `GOAL_STEP_FROM_RVIZ=1`
- RViz tools:
  - "Publish Point" publishes `geometry_msgs/PointStamped` on `/clicked_point`
  - "2D Nav Goal" publishes `geometry_msgs/PoseStamped` on `/move_base_simple/goal`
- The helper subscribes to both topics and stores the latest goal transformed into `torso_lift_link`.

Implementation: `tiago_client/tiago_client/utils/rviz_goal_listener.py`

## Architecture

- **`intent_predictor_integrated.py`**: The brain. Runs the VLM loop, analyzes geometry, and calculates confidence.
- **`tiago_client.py`**: The body. Receives VR inputs, mixes them with the "Assistance Force" from the predictor, and sends safe commands to the robot.
- **`run_tiago_real.py`**: The coordinator. extracting state (Images, Obstacles, EE Pose) and passing it between the Client and Predictor.

## Predefined Tabletop Obstacles (Offline)

See `tiago_server/tiago_server/perception/README_predefined_spheres.md` for recording and replaying `/detected_spheres` from a predefined tabletop environment.
