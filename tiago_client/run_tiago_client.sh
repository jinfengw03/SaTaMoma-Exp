source /home/jiachenli/miniconda3/bin/activate
conda deactivate
source /opt/ros/noetic/setup.bash
conda activate ros39
export TIAGO_TELEOP_TYPE=KEYBOARD
python tiago_client/run_tiago_real.py