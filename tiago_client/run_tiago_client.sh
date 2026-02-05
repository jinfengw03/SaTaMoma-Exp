#!/bin/bash
# 1. Activate base conda to use 'conda' commands
source /home/jiachenli/miniconda3/bin/activate

# 2. Reset and activate the robot-specific environment
conda deactivate
source /opt/ros/noetic/setup.bash
conda activate ros39

# 3. Set existing robot teleop variablesss
export TIAGO_TELEOP_TYPE=KEYBOARD
export ENABLE_VLM=${ENABLE_VLM:-1}  # Set to 1 to enable VLM, 0 to disable (default)
python tiago_client/run_tiago_real.py