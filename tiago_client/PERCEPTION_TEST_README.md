# Perception System Testing Guide

This guide details how to verify the integration of the **Vision Perception Module** (`pointcloud_to_sphere.py`) with the TIAGo Server/Client architecture.

## 1. Prerequisites (Server Side)

Before running the server, ensure the robot's environment has the necessary calculation libraries.

**On the Robot:**
```bash
# Activate the environment used by the server
source /home/pal/miniconda3/bin/activate
conda activate tiago

# Install required libraries for clustering and fitting
pip install scikit-learn scipy
```

## 2. Launching the Server

The perception node is now integrated into the server startup script.

**On the Robot:**
```bash
cd ~/SaTaMoma-Exp
bash tiago_server/run_tiago_server.sh
```

**Verify Startup:**
Check the terminal output for the following line, indicating the background perception process has started:
```text
Starting PointCloud to Sphere Node...
```

## 3. Verification

### Method A: Client Monitor Script (Recommended)
We have created a lightweight tool to inspect the data stream from the server without starting the full VR teleoperation.

**On the Client (Local Machine):**
```bash
# Ensure you are in the root of the workspace
cd ~/SaTaMoma-Exp

# Run the verification script
python tiago_client/verify_perception.py
```

**Expected Output:**
You should see a continuously updating table. If you place an object in front of the robot (within 0.8m), values should appear:
```text
--- Perception Monitor (Updated: 14:30:05) ---

[+] Detected 2 spheres (torso_lift_link frame):
ID   X (m)      Y (m)      Z (m)      Radius (m)
--------------------------------------------------
0    0.650      -0.100     0.800      0.055     
1    0.660      -0.080     0.810      0.030     
```

### Method B: RViz Visualization
If you have a ROS environment connected to the robot's master, you can see the debugging markers directly.

1.  Open RViz (`rosrun rviz rviz`).
2.  Set **Fixed Frame** to `torso_lift_link` or `base_footprint`.
3.  Add **MarkerArray** display -> Topic: `/sphere_markers`.
4.  Add **PointCloud2** display -> Topic: `/camera_pointcloud`.

*You should see red spheres enclosing the obstacles.*

## 4. Troubleshooting

| Symptom | Possible Cause | Solution |
|---------|----------------|----------|
| **Script says "Key 'obstacles' not found"** | Server code outdated | Update `tiago_server/tiago_server/tiago_server_node.py` on the robot. |
| **"No obstacles detected"** | Objects too far | The depth filter is set to **< 0.87m**. Move objects closer. |
| **"No obstacles detected"** | Perception node crashed | Check server terminal for python crash logs (e.g. missing `sklearn`). |
| **Spheres appear inside robot arm** | Self-filtering failed | Check if `torso_lift_link` TF is correct or increase overlap threshold. |

---
*Note: The perception node runs asynchronously in the background. If you stop `run_tiago_server.sh` with Ctrl+C, the script attempts to kill the background process automatically.*
