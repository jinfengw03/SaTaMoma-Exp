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

# Start the pointcloud to sphere perception node in the background
echo "Starting PointCloud to Sphere Node..."
python tiago_server/perception/pointcloud_to_sphere.py > /dev/null 2>&1 &
PERCEPTION_PID=$!

#run node
python /home/pal/krhinok/SaTaMoma-Exp/tiago_server/tiago_server/tiago_server_node.py

# Kill perception node when server exits
kill $PERCEPTION_PID