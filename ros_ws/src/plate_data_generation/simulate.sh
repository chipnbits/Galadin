#!/bin/bash

NUM_RESETS=800
SLEEP_BEFORE_KILL=18
SLEEP_BEFORE_RESTART=25

for ((i=1; i<=$NUM_RESETS; i++)); do
    echo "Starting simulation"
    python3 ~/ros_ws/src/2022_competition/enph353/enph353_gazebo/scripts/plate_generator.py &
    ~/ros_ws/src/2022_competition/enph353/enph353_utils/scripts/run_sim.sh -g &
    pid=$!
    sleep $SLEEP_BEFORE_KILL
    echo "Killing simulation"
    pkill -f sim.launch
    sleep $SLEEP_BEFORE_RESTART
done