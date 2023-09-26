#!/bin/bash

FIRST_IMAGE_TOPIC=/R1/fixate_vision # Change this to your desired topic
SECOND_IMAGE_TOPIC=/R1/pi_camera/image_raw
THIRD_IMAGE_TOPIC=/R1/machine_vision_experiments

# Define a function for starting the simulation and running it for 60 seconds
function run_simulation() {
    echo "Starting simulation script"
    python3 ~/gadalin/ros_ws/src/2022_competition/enph353/enph353_gazebo/scripts/plate_generator.py &
    ~/gadalin/ros_ws/src/2022_competition/enph353/enph353_utils/scripts/run_sim.sh -vpg &
    pid=$!

    sleep 5 && rosrun rqt_image_view rqt_image_view $FIRST_IMAGE_TOPIC &
    sleep 5 && rosrun rqt_image_view rqt_image_view $SECOND_IMAGE_TOPIC &  
    sleep 5 && rosrun rqt_image_view rqt_image_view $THIRD_IMAGE_TOPIC & 

    # Register a signal handler for Ctrl+C that kills all child processes
    function cleanup() {
        echo "Cleaning up..."
        kill -- -$pid  # Sends SIGTERM to the process group of the main process
        pkill -f gazebo  
    }
}

run_simulation

trap cleanup SIGINT

# Wait for child processes to finish
wait
