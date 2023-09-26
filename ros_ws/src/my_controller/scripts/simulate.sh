#!/bin/bash

NUM_RESETS=800
SLEEP_BEFORE_KILL= 59
SLEEP_BEFORE_RESTART=25
FIRST_IMAGE_TOPIC=/R1/fixate_vision # Change this to your desired topic
SECOND_IMAGE_TOPIC=/R1/pi_camera/image_raw
SCORE_TRACKER_SCRIPT=~/gadalin/ros_ws/src/2022_competition/enph353/enph353_utils/scripts/score_tracker.py 

for ((i=1; i<=$NUM_RESETS; i++)); do
    echo "Starting simulation script"
    python3 ~/gadalin/ros_ws/src/2022_competition/enph353/enph353_gazebo/scripts/plate_generator.py &
    ~/gadalin/ros_ws/src/2022_competition/enph353/enph353_utils/scripts/run_sim.sh -vpg &
    pid=$!
    echo "Launching controller"
    sleep 9 && roslaunch my_controller robot_controller.launch &
    sleep 5 && rosrun rqt_image_view rqt_image_view $FIRST_IMAGE_TOPIC &
    sleep 5 && rosrun rqt_image_view rqt_image_view $SECOND_IMAGE_TOPIC &
    sleep 5 && gnome-terminal --tab -e "python3 $SCORE_TRACKER_SCRIPT" &
    sleep $SLEEP_BEFORE_KILL
    echo "Killing simulation"
    pkill -f sim.launch
    pkill -f robot_controller.launch	
    pkill -f rqt_image_view	
    pkill -f "$SCORE_TRACKER_SCRIPT" # Kill the score tracker script
    sleep $SLEEP_BEFORE_RESTART
done
