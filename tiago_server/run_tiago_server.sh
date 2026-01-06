# activate conda environment
source /home/pal/miniconda3/bin/activate
conda activate tiago

# export env vars
export PAL_DISTRO=gallium
export ROS_DISTRO=noetic
source /opt/pal/${PAL_DISTRO}/setup.bash
export ROS_MASTER_URI=http://tiago-224c:11311
export ROS_IP=192.168.0.110
cd tiago_server
export PYTHONPATH=$PYTHONPATH:$(pwd)

#run node
python /home/pal/krhinok/SaTaMoma-Exp/tiago_server/tiago_server/tiago_server_node.py