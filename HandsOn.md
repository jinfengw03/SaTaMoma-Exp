# Hands on instruction for experiment
## Hardware setup procedure
### TIAGo
- There are two buttons in the back of TIAGo. Press left then right to launch. Right button is for the control PC, press it first when shutting down. 
- You can access the command panel using the dev PC's browser. Execute the **Home** command when shutting down.
- TIAGo doesn't lower its torso after you press the right button when shutting down, you need to hold its arm(recommend holding the end-effector) to avoid collision with the ground
- The connection between the dev PC and the TIAGo is already set up. If not, refer to the [script](tiago_server/run_tiago_server.sh) for environment configuration.
### Network
- Connect inference PC to the router using cables or WiFi. This is for the communication with the Meta Quest and the Dev PC
### Meta Quest
- Make sure the inference PC is connected to the router. Put on Meta Quest and launch the client(remember to launch server first), which will attempt to connect with the VR device. The first connection attempt will fail and you need to agree the USB debugging in the VR, launch again and it should work.

## Software setup procedure
### Server PC(Dev PC)
- Use the conda environment called tiago, which is already setup on the dev PC. Then bash the [script](tiago_server/run_tiago_server.sh), which will setup the env variables and launch the server automatically. After the arms are in place, you will see a line suggesting ENTER. Press and it's done.
### Client PC(Inference PC)
- Launch terminal. Make sure no conda envrionment is activated, if any, please deactivate. Then execute
```
source /opt/ros/noetic/setup.bash
```
- Activate conda envrionment **ros39** and 
```
python tiago_client/run_tiago_real.py
```

## Extra things to look out for
- We do not use the absolute pose of the two touch controllers as the arms' pose. We calculate the relative movement of the touch controllers and then add to the current pose of the arm, then solve the inverse kinematics. Keep this in mind when teleoperate and try to find the trick.
- You need to put the headset on when launching the client. You may prop the headset up on your forehead to see clearer when everything is ready.
- When you teleoperate, remember to stay within the certain circle.(_You can try to walk out of the circle and you will know how the circle looks like_) You can also walk out the circle to redefine your working space.
- TIAGo's arms have limited working space. The initial position for the arms are best for you to teleoperate on, try to stay in the neighboring space when teleoperate. The accesible working space should be enough for tasks like picking and placing. 
- If you unfortunately moved the arm out of its workable space, that is when you see ik solver keeps failing for the arm you are teleoperating, you may need to shut down the client then the server and restart.
- TIAGo's base is not easy to control when it makes turns.