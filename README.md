# gadalin
UBC ENPH 353 - CNN Robot Vision project course

## File description of repo

- `ros_ws/`: Contains the ROS workspace.
    - `build/`: Contains the build files for ROS packages.
    - `devel/`: Contains the compiled binaries for ROS packages.
    - `src/`: Contains the source code for ROS packages.
    
      - `my_controller`: Our robot controller
        - `launch/`: Contains the launch file to launch the plate detection and controller nodes in parallel
        -  `nodes/`: Python files for implementing the control algorithms
            - `robot_controller` : The main robot controller.
            - `plate_detection.py`: A seperate node for automated plate gathering.
            - `vision_processing` : Helper functions for machine vision across all components of the competition.
            - `inner_loop_PID.py` : Python class object for PID control agent in the inner loop of the course.
         - `saved_images` : Directory for saving snapshots to.  
         - `scripts/` : Bash scripts for automation of testing and verification and secondary plate gathering
        
        - `plate_data_generation` : A package from early development for gathering plates overnight at still positions
          - `launch` : Contains launch file for the package
          - `nodes/plate_data_snapshots.py : A script for automated capture of plates from the environment after launch
          - `robots_many.launch` : Launch file to replace robots.launch in the competition package for plate gathering with 8 robots
          - `simulate.sh` : Bash script for automation of relauching ROS overnight for plate gathering

        - `2022_competiton` : The competition environment provided for the course

- `cnn_trainer/`: Contains the code and training data for making CNNs used in the competition.
  - `cnn_alpha/` : CNN for reading plate characters.
    - `placards/` : The final training data set that was used for plate character recognition.
    - `wandb/` : Training data for the final models that were implemented.
    - `weights/` : Saved weights from each of the training epochs of the last trained model.
    - `alphachar_image_processor.ipynb`: Notebook for training.
    - `model.json`: A trained model that is saved
    - `model.h5`:  Saved weights and parameters to accompany model
  - `cnn_parking_numbers/` : CNN for reading the parking IDs off the palcards
    - `parking_image_processor.ipynb`: Notebook for training.
    - `model.json`: A trained model that is saved
    - `model.h5`:  Saved weights and parameters to accompany model




## Useful aliases for the competition source

```bash
alias teleop='rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=R1/cmd_vel'
alias camfeed='rosrun rqt_image_view rqt_image_view'
alias edit_source='nano ~/ros_ws/devel/setup.zsh'
alias runsim='~/ros_ws/src/2022_competition/enph353/enph353_utils/scripts/run_sim.sh -vpg'
alias robot_controller='roslaunch my_controller robot_controller.launch'
alias runsim_photomode='~/ros_ws/src/2022_competition/enph353/enph353_utils/scripts/run_sim.sh -g'
alias plate_gen='python3 ~/ros_ws/src/2022_competition/enph353/enph353_gazebo/scripts/plate_generator.py'
```

Here are some useful ROS commands for the competition:

- `teleop`: Turns on keyboard control (I turned mine off by default)
- `camfeed`: Opens an image topic viewing panel to view multiple topics at once
- `edit_source`: Quickly opens the competition source file to make changes like adding aliases
- `runsim`: Boots up Gazebo and runs the simulation in regular mode
- `runsim_photomode`: Runs the simulation without vehicles and pedestrians for working on machine vision
- `robot_controller`: Runs the launch file in the `my_controller` package
- `plate_gen`: Manually regenerates plates (not working through the default repo's `run_sim` launch)

To use these commands, you can simply type them into your terminal or add them to your `.bashrc` file for quick access.



## Plate Detection Data Flow

![plate_filtering drawio](https://i.imgur.com/FmMI6pr.png)


