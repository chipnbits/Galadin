#!/usr/bin/env python3

import os
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math

from datetime import datetime

import vision_processing as vision
import inner_loop_PID as inner_loop

# Import the PID controller
# https://pypi.org/project/simple-pid/
from simple_pid import PID

class RobotController(object):
    def __init__(self):

        self.bridge = CvBridge()
        # Store image from subscribed topic
        self.image = None
        self.last_image = None # used for differential image processing
        # Store the last time an image was captured from image_raw
        self.last_capture_time = None
        # Store the last time an image was processed and published
        self.last_pub_time = None
        # Sets the frame rate for updating image topics
        self.frame_out_interval = .002 # seconds
        
        # Subscribe to the camera image and joystick topics (gives camera feed and direct access to controller)
        self.image_sub = rospy.Subscriber('R1/pi_camera/image_raw', Image, self.callback_image)
        # Joy controller - Reads commands for manual driving and debugging
        self.joy_sub = rospy.Subscriber('R1/joy', Joy, self.callback_joy)        
        # Create a subscriber and field to the /clock topic
        self.clock_sub = rospy.Subscriber('/clock', Clock, self.callback_clock)
        # Store the time from the /clock topic - used for simulation time tracking on hardcode
        self.clock = None 
        

        # Subscribe to cmd_vel topic to send control commands to the robot
        self.cmd_vel_pub = rospy.Publisher('R1/cmd_vel', Twist, queue_size=20)
        # Create a publisher to the /license_plate topic to communicate with Score Tracking
        self.license_plate_pub = rospy.Publisher('/license_plate', String)    
        # Create a publisher and subscriber for the kill command to communicate termination between plate node and controller
        self.kill_pub = rospy.Publisher('kill_command', String, queue_size=10)
        self.kill_sub = rospy.Subscriber('kill_command', String, self.callback_kill)    
        #Publish line processing vision to the line_processing topic for debugging
        self.image_line_processing_pub = rospy.Publisher("R1/line_processing", Image)

        # Storage for processed images
        self.image_line_message_out = None

        #Publish the processed image from fixate (the sky box algorithm ot align with end of road
        self.fixate_pub = rospy.Publisher("R1/fixate_vision", Image)
        #Publish a machine vision experiment image, this is a topic for prototyping machinevision
        self.experimental_vision_pub = rospy.Publisher("R1/machine_vision_experiments", Image)
        
        # Create a directory to save images captured with the controller button
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved_images', 'raw'))
        os.makedirs(self.output_dir, exist_ok=True)
        # Add essentially a debounce on the button press (limit to 1 picture per 2 seconds)
        self.capture_interval = 0.2 # seconds

        # State machine state, see control loop for details
        # Manual control achieved setting this to -1 sentinel value
        # Regualar autonomous control with 0
        self.state = 0
        # Pedestrian crossing Interrupt flag
        self.interrupt = False        
        # PID vals are overwritten in the fixate function
        self.fixate_controller = PID(.05, 0, 0, setpoint=0)
        self.TRUCK_THRESHOLD = 50   #550 is leaving a far distance for truck, 1700 is closer
        self.consecutive_readings_below_threshold = 0 # Consecutive truck readings

        # Stop robot signal from palte detect
        self.kill_switch = False

    def initiate_controller(self, delay):
        '''
        @brief Delay start of controller to allow instantiation of fields

        @param delay Delay time in seconds

        @retval None

        '''
        # Allow for a delay before starting the controller
        print(f'Delaying start for {delay} seconds')
        rospy.sleep(delay)
        print("Starting controller")

    def callback_image(self, data):
        '''
        @brief Process incoming image into machine vision at a set interval determined by self.frame_out_interval

        This function converts the topic image to OpenCV format, changes the color space to RGB,
        and stores it internally as a field accessible from other functions.
        Debug mode publishes to the fixate and line follow topics for viewing

        @param data 'R1/pi_camera/image_raw': The input image from the ROS topic.

        @retval None

        '''
        # Process incoming image if it has been long enough since the last image was processed
        now = rospy.Time.now()
        if self.last_pub_time is None or (now - self.last_pub_time).to_sec() > self.frame_out_interval:
            # Convert the topic image to OpenCV format
            img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            
            # Convert the image to RGB color space (the color orderings are different between OpenCV and ROS)
            # This critically updates self.image which is used by machine vision in other functions of the controller
            self.last_image = self.image
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Call pedestrian crossing detection at all times
            # If a crossing is detected, stops robot and indicates to controller 
            # Through the self.interrupt field
            self.ped_crossing_flag()

            # Debug mode, when manual control is selected (self.state = -1)
            if (self.state == -1):
                # Publish the image debug for fixate
                scan, extension_pixel = vision.skyScanner(self.image)
                if(scan is not None):
                    self.publish_to_fixate(scan)

                _,img = self.truck_clear()       
                if(img is not None):
                    self.publish_to_experimental_vision(img)
                
                # Publish the image debug for line recognition
                self.line_recognition()
                if(self.image_line_message_out is not None):
                    self.image_line_processing_pub.publish(self.image_line_message_out)

            # Log the last time an image was published
            self.last_pub_time = rospy.Time.now()

    def callback_joy(self, data):
        """
        @brief Take a screenshot of the raw image with timestamp and save it to a folder.

        This function checks if the joystick button is pressed and if an image is available. If both conditions are true,
        the function takes a screenshot of the image, adds a timestamp to the filename, and saves it to a specified output
        directory.

        @param data A Joy message containing the joystick state.
        @return None
        @exception None
        """

        # Take snapshots with the joy button
        if data.buttons[0] == 1: # change 0 to the button index you want to use, 
                                # pulls directly from the button array in joy topic
            if self.image is not None:  # Check if an image is available
                now = rospy.Time.now()  # Debounce Check
                if self.last_capture_time is None or (now - self.last_capture_time).to_sec() > self.capture_interval:
                    # If enough time has passed since the last capture, take a new screenshot

                    timestamp = datetime.now().strftime("%m%d-%H%M%S")  # Add a timestamp to the filename
                    filename = os.path.join(self.output_dir, f'image_{timestamp}.png')  # Construct the full path of the output file
                    cv2.imwrite(filename, self.image)  # Save the image to disk
                    rospy.loginfo('Image saved to {}'.format(filename))  # Log the filename to the ROS log
                    # Print message to console
                    print(f"Image saved to {filename}")
                    self.last_capture_time = now
                else:
                    # If not enough time has passed, skip this capture
                    pass
            else:
                # If no image is available, log a message to the ROS log
                rospy.loginfo('No image received yet')

    def callback_kill(self, msg):
        '''
        @brief Determine if the plate reading is finished (Last plate read)

        @param msg a string containing information from the topic

        @retval None

        '''
        # If the message is 'kill', shut down the driving
        if msg.data == 'kill':
            self.kill_switch = True

    def control_loop(self):
        '''
        @brief Driving control loop for the competition. Includes various states to allow for checkpointing

        State -1: Debug and manual control, will bypass automatic control sequence
        State  0: From start position, read first plate, execute first turn and read 2nd plate
        State  1: Drive as fast as possible to first pedestrian crossing
        State  2: Wait for first pedestrian to clear then move forward
        State  3: Complete second turn, drive to 2nd pedestrian
        State  4: Wait for 2nd pedestrian to clear, then drive forward 
        State  5: Maneuver through 1st dirt turn, read plate 4, drive dirt, read plate 5,
                  take 2nd dirt turn, read plate 6, return to start point
        State  6: Not in use
        State  7: Run truck detection and wait, then enter loop for handover to PID control
        State  8: Run the PID line follow agent for a set period to get last 2 plates

        @retval None
        '''

        print("In state", self.state)

        if self.state == -1:         
            pass

        elif self.state >= 0:

            if self.state ==0:
                # Enter outer track, turn left head into first left turn
                # Read both plates in routine
                self.turn_and_drive_robot(distance = .81, speed = .5 )
                rospy.sleep(.5)
                self.turn_and_drive_robot(distance = .15, speed = .4 )
                self.drive_forward(.52,1)
                self.turn_and_drive_robot(distance = .55, speed = .7 )
                self.turn_robot(33.5, 2.4)
                self.turn_robot(-33, 2.4)
                self.turn_and_drive_robot(distance = .32, speed = .6 )
                self.fixate(setpoint = 641)            
                self.state = 1

            if self.state == 1:
                print("In state", self.state)
                # Drive fast to pedestrian                                              
                self.drive_forward(.1,.5)
                self.drive_forward(.1,1.1)
                self.drive_forward(.7,1.6)
                self.drive_forward(.1,1.1)
                self.drive_forward(.1,.7)
                # Slow down before crosswalk
                self.drive_forward(1,.5)


            if self.state == 2:
                print("Pedestrian detected, now in state", self.state)                
                # Stop the robot
                self.stop_robot()            
                # Check if it's the right time to move forward, and update the state accordingly
                if (self.pedestrian_clear()):
                    print("Pedi is clear, going to state 3")
                    # Drive clear of pedestrian
                    self.drive_forward(.1,.6)
                    self.drive_forward(.46,1.0)
                    self.state = 3            

            if self.state == 3:
                print("In state", self.state)  
                # Complete second turn with a plate check
                self.turn_and_drive_robot(distance = .55, speed = .7 )
                self.turn_robot(33.5, 2.4)
                self.turn_robot(-37, 2.4)
                self.turn_and_drive_robot(distance = .325, speed = .6 )
                self.fixate(setpoint= 640)
                # Drive fast to 2nd pedestrian
                self.drive_forward(.1,.5)
                self.drive_forward(.1,1.1)
                self.drive_forward(.6,1.6)
                self.drive_forward(.1,1.1)
                self.drive_forward(.25,.7)
                # Slow down before crosswalk
                self.drive_forward(1,.4)
                print("Pedestrian detected, from state 3")

            # At 2nd pedestrian
            if self.state == 4:
                print("In state", self.state)                
                # Stop the robot
                self.stop_robot()
                # Check if it's the right time to move forward, and update the state accordingly
                if self.pedestrian_clear():
                    print("Pedestrian clear, entering state 5")
                    # Drive clear of pedestrian
                    self.drive_forward(.1,.5)
                    self.drive_forward(.50,.9)
                    self.state = 5
            
            if self.state == 5:
                print("In state", self.state)

                # First dirt corner with plate check
                self.turn_and_drive_robot(distance = .55, speed = .6 )
                self.turn_robot(33.5, 2.2)
                self.turn_robot(-37, 2.2)
                self.turn_and_drive_robot(distance = .29, speed = .5 )
                # Bias the setpoint  low to turn more RIGHT 640 is middlee
                self.fixate(setpoint = 634)  

                # Blast through dirt track
                self.drive_forward(.08,.45)
                self.drive_forward(.72,.9)
                # Detour to snap P5 plate
                self.turn_and_drive_robot(distance = .26, speed = .8 )
                self.turn_and_drive_robot(distance = -.36, speed = .20 )
                # Finish straight dirt section
                self.drive_forward(.1,.3)
                self.drive_forward(.42,1.0)
                self.drive_forward(.12,.5)

                # Take 4th corner (2nd dirt corner)
                self.turn_and_drive_robot(distance = .65, speed = .6 )
                self.turn_robot(33.5, 2.4)
                self.turn_robot(-40, 2.4)
                self.turn_and_drive_robot(distance = .24, speed = .6 )
                self.fixate(setpoint= 640)

                # Drive to inner loop entrance
                self.drive_forward(.1,.5)
                self.drive_forward(.37,.7)
                self.drive_forward(.11,.5)
                # Enter inner loop
                self.turn_and_drive_robot(distance = .9, speed = .6 )
                self.stop_robot()
                self.state = 6

            # Spaceholder not used
            if self.state == 6:
                print("In state", self.state)
                self.state = 7

            if self.state == 7:
                # Check for truck clearance
                # Check the second argument of the function.  First arg is img, second is bool
                isClear, _ = self.truck_clear()
                while (isClear == False):
                    isClear, img = self.truck_clear()
                    self.publish_to_experimental_vision(img)
                    rospy.sleep(.05)
                
                print("Truck Cleared entering state 8")                
                self.state = 8

            if self.state == 8:
                self.turn_and_drive_robot(distance = .2, speed = .7 )

                #Run inner loop agent for fixed time to get to plate 8
                agent = inner_loop.PIDAgent()
                rospy.sleep(5.5)  #9.5 and PID at .32 speed was good combo
                                   #8.98 and PID at .35 speed was good combo

                # Shutdown the agent 
                agent.stop_agent()
                del agent
                self.stop_robot()

                # If early finish of 4 positive P8 matches, then skip this
                # Otherwise we force a good vantage
                if self.kill_switch == False:
                    # Force a good vantage for last plate
                    # Plate 8 is worse than all others to read
                    self.turn_robot(-60,4)
                    self.stop_robot()
                    self.drive_forward(.18,.5)
                    rospy.sleep(.5)

                # Turn off timer and wait for rear-ending with truck
                self.stop_robot()
                # Kill plate reader
                self.kill_pub.publish('kill')

                # Wait for finishing position to be assesed
                rospy.sleep(8)

                # victory dance (super spin)
                twist = Twist()
                twist.angular.z = 30  #prev .15
                # Publish the turning command
                self.cmd_vel_pub.publish(twist)
                rospy.sleep(1)
                twist = Twist()
                twist.angular.z = 42  #prev .15
                # Publish the turning command
                self.cmd_vel_pub.publish(twist)
                rospy.sleep(10)
                # Stop the whirlwind
                self.stop_robot()

    def fixate(self, setpoint = 640):
        '''
        @brief Fixes the skyscanner end-of-road to a setpoint, default is 640

        The function rotates the robot to get a precise alignment with 
        the other end of a road segment

        @param setpoint: The pixel value to align with the skyScanner, 
        a value less than 640 will turn the robot further left, 
        and a value greater than 640 will turn the robot further right

        @retval None
        '''
        curr_point = 0        
        
        self.fixate_controller.setpoint = setpoint
        self.fixate_controller.tunings = (.032, 0, .001)

        #Important to set the output limits to avoid extreme overshoot
        self.fixate_controller.output_limits = (-.5, .5)

        twist = Twist()
        twist.angular.z = 0 
  
        rate = rospy.Rate(35) 
        # Do PID fixation while target is still not reached
        while True:
            scan, extension_pixel = vision.skyScanner(self.image)

            if extension_pixel is not None:
                curr_point = extension_pixel[0]

            control_value = self.fixate_controller(curr_point)

            twist.angular.z = control_value  # prev .15
            # Publish the turning command
            self.cmd_vel_pub.publish(twist)

            if scan is not None:
                self.publish_to_fixate(scan)

            if abs(setpoint - curr_point) <= 3:
                consecutive_valid_readings += 1
            else:
                consecutive_valid_readings = 0

            if consecutive_valid_readings >= 6:
                break

            rate.sleep()  # Sleep to enforce the desired publishing rate

            # Debugging
            #print(curr_point,setpoint)

        # Stop the rotation by setting the angular velocity to zero
        twist.angular.z = 0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(.05)
        self.cmd_vel_pub.publish(twist)
       
    def turn_robot(self, angle_degrees, turning_speed=.3):
        """
        @brief Turn the robot a specified angle amount at a specified turning speed.

        @param angle_degrees: The angle in degrees to turn the robot. Positive angles turn counterclockwise, negative angles turn clockwise.
        @param turning_speed: (optional) The turning speed in radians per second. Default is 0.3.

        @return None
        """
        
        # TODO: Remove the math library import and do this one operation manually
        # Convert angle to radians
        angle_rad = math.radians(angle_degrees)

        # Calculate the duration based on the turning speed
        duration = abs(angle_rad / turning_speed)*1.415

        # Create a Twist message for the turning command
        twist = Twist()
        twist.angular.z = turning_speed if angle_degrees >= 0 else -turning_speed

        # Publish the turning command
        self.cmd_vel_pub.publish(twist)
        
        # Wait for the calculated duration with the option to interrupt (Pedestrian detection)
        start_time = self.clock
        while (self.clock - start_time).to_sec() < duration:
            if self.interrupt:
                break

        # Stop the rotation by setting the angular velocity to zero
        twist.angular.z = 0
        self.cmd_vel_pub.publish(twist)

    def drive_forward(self, distance, speed=.3):
        """
        @brief Drive the robot forward for a specified distance at a specified speed.

        @param distance: The distance roughly in meters to drive the robot. Positive values move forward, negative values move backward.
        @param speed: (optional) The driving speed in meters per second. Default is 0.3.

        @return None
        """

        # Calculate the duration based on the distance and speed
        duration = abs(distance / speed)

        # Create a Twist message for the driving command
        twist = Twist()
        twist.linear.x = speed if distance >= 0 else -speed

        # Publish the driving command
        self.cmd_vel_pub.publish(twist)

        # Wait for the calculated duration with the option to interrupt Pedestrian detection
        start_time = self.clock
        while (self.clock - start_time).to_sec() < duration:
            if self.interrupt:
                break
            rospy.sleep(0.05)

        # Stop the robot by setting the linear velocity to zero
        self.stop_robot()
    
    def turn_and_drive_robot(self, distance=1, speed=.4):
        """
        @brief Turn and drive the robot in an arc for a specified distance at a specified speed.

        @param distance: The distance in meters to drive the robot in an arc. Positive values turn left, negative right
        @param speed: (optional) The driving speed in meters per second. Default is 0.4.

        @return None
        """
        # Calculate the duration based on the distance and speed
        duration = abs(distance / speed)*.6

        # Create a Twist message for the driving command
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = speed*3.4 if distance >= 0 else -speed*3.4

        # Publish the driving command
        self.cmd_vel_pub.publish(twist)

        # Wait for the calculated duration with the option to interrupt
        start_time = self.clock
        while (self.clock - start_time).to_sec() < duration:
            if self.interrupt:
                break
            rospy.sleep(0.01)

        # Stop the robot by setting the linear velocity to zero
        self.stop_robot()

    def stop_robot(self):
        """
        @brief Stop the robot by setting the linear and angular velocity to zero.

        @return None
        """
        twist = Twist()
        twist.linear.x = 0
        twist.angular.z = 0
        self.cmd_vel_pub.publish(twist)

    def ped_crossing_flag(self):
        """
        @brief Scans for a red stop line signature of pedestrian crossing and flags the robot

        The red line is detected by HSV filtering on the red color and detecting a contour that is over a threshold

        @modifies self.state Moves robot into next state if found
                  self.interrupt sends an interrupt message to controls to stop robot

        @return None

        """
        # Convert the image to the HSV color space
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Define the red color range in HSV
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])

        # Create a mask for the red color
        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        # Find contours in the masked image
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the area of the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)

            # Print a warning if the area is above a certain threshold
            area_threshold = 12000
            if largest_area > area_threshold:
                #print(f"Warning: Large red object detected. Area: {largest_area}")
                if(self.state == 1):
                    print("First Pedestrian detected, stopping robot")
                    self.state = 2
                    self.interrupt = True
                if (self.state == 3):
                    print("Second Pedestrian detected, stopping robot")
                    self.state = 4
                    self.interrupt = True

    def get_ped_differences(self):
        """
        @brief COmpares two consecutive frames of the pedestrian and returns the number of pants pixesl that have changed

        The pedestrian pants are filtered with hsv, then two consecutive camera frames are compared bitwise XOR to determine 
        if the pedestrian is moving or about to cross


        @return diff an integer representation of the number of changed pixels
        @exception None
        """
        # Filter the previous frame for the pedestrian pants
        hsv = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2HSV)
        # Cutout the the image to focus on pedestrian
        height, width = hsv.shape[:2]
        hsv =  hsv[300:(height-100), 150:width-150]
        last_img = vision.filterPedPants(hsv)

        # Filter the current frame for the pedestrian pants
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # Cutout the the image to focus on pedestrian
        height, width = hsv.shape[:2]
        hsv =  hsv[300:(height-100), 150:width-150]
        img = vision.filterPedPants(hsv)

        # Find the difference between the two images
        diff = cv2.absdiff(last_img, img)

        return diff   

    def pedestrian_clear(self):
        """
        @brief Determine if the pedestrian is clear based on image differences.

        Takes the difference between the last image and the current image, filters them both for the 
        Pedestrian pant signature.  The pedestrian is still if not about to cross.  Does a bitwise or 
        on the two images to find the difference between the two.  If the difference is below a threshold,
        the pedestrian is considered clear.

        @return bool Returns True when the pedestrian is considered clear.
        """

        # Quantify the difference in frames by pixel count
        motion_value = cv2.countNonZero(self.get_ped_differences())           

        # Wait for pedestrian to be still to go
        while (motion_value > 15):
            rospy.sleep(self.frame_out_interval)
            motion_value = cv2.countNonZero(self.get_ped_differences())

        print("Pedestrian clear")
        self.interrupt = False

        return True

    def truck_clear(self):
        """
        @brief Filter for the truck and determine if it is not in conflict with entry to inner loop, produce debug image

        The truck is thresholded based on the black color of the windows and tires.  The top right of the image is ignored, since it is a clearance
        area.  The total number of truck threshold pixels is counted and if the truck image is small enough, robot is cleared for entry 
        after 3 consecutive clear readings. The debug image shows the threshold view and prints a Red Light for nogo, Yellow for below threshold but
        not yet 3 consecutive readings, a green light for go 

        @param takes in images from the self.image field
        @return boolean - truck clearance status
                img - a cv2 image of the debugging view
        @exception None
        """
        
        # Trim the image to bias viewing to the left
        _, width = self.image.shape[:2]
        img = vision.thresholdTruck(self.image[:, :width-210])

        # Quantify the amount of truck in the frame by pixel count
        reading = cv2.countNonZero(img)

        print(reading)

        # Mark with a dot for stop or go based on a threshold
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if reading > self.TRUCK_THRESHOLD:
            # Reset the consecutive readings counter
            self.consecutive_readings_below_threshold = 0

            # Place a red dot on the image
            cv2.circle(img, (50, 50), 50, (0, 0, 255), -1)
            return False, img
        else:
            # Increment the consecutive readings counter
            self.consecutive_readings_below_threshold += 1

            # Check if the consecutive readings counter reached 3
            if self.consecutive_readings_below_threshold >= 3:
                # Place a green dot on the image
                cv2.circle(img, (50, 50), 50, (0, 255, 0), -1)
                return True, img
            else:
                # Place a yellow dot on the image to indicate not enough consecutive readings
                cv2.circle(img, (50, 50), 50, (0, 255, 255), -1)
                return False, img
   
    def line_recognition(self):
        """
        @brief Recognizes the road surface with hsv filtering. publishes the thresholded image for debugging, used for 
        debugging of the inner loop agent when in debugging mode (state = -1)

        @modifies self.image_line_message_out
        @param self.image the most recent image from the camera

        @return None
        """

        mask = vision.lineVision(self.image)       
        self.image_line_message_out = self.bridge.cv2_to_imgmsg(mask, "mono8")

    def publish_to_fixate(self, img):
        """
        @brief Takes a cv2 image and publishes it to the fixate rostopic

        @param img the image to be published

        @return None
        """

        if (img is None):
            return
        try:
            if len(img.shape) == 2:  # Grayscale image
                msg_out = self.bridge.cv2_to_imgmsg(img, "mono8")
            else:  # Color image
                msg_out = self.bridge.cv2_to_imgmsg(img, "bgr8")
        except CvBridgeError as e:
            print(e)

        if msg_out is not None:
            self.fixate_pub.publish(msg_out)

    # Publish experiment of vision for trying new ideas
    def publish_to_experimental_vision(self, img):
        """
        @brief Takes a cv2 image and publishes it to the experimental vision rostopic

        This function is used as a general auxilary channel to publish any image data for debugging

        @param img the image to be published

        @return None
        """
        if (img is None):
            return
        try:
            if len(img.shape) == 2:  # Grayscale image
                msg_out = self.bridge.cv2_to_imgmsg(img, "mono8")
            else:  # Color image
                msg_out = self.bridge.cv2_to_imgmsg(img, "bgr8")
        except CvBridgeError as e:
            print(e)

        if msg_out is not None:
            self.experimental_vision_pub.publish(msg_out)

    def callback_clock(self, data):
        """
        @brief Callback function for the clock topic

        Used to update the internal clock variable that is used to keep track of time for hardcode sequences.

        @param data A ROS message containing the current time.
        @return None
        @exception None
        """

        self.clock = data.clock    

    def start_timer(self):
        '''
        @brief Start the timer for the competition.

        Sends a message on the /license_plate topic with license plate ID 0 (zero) to start the timer.

        @retval None
        '''

        # Team info
        team_id = "Gadalin"
        team_password = "bowser"
        plate_id = "PA55"

        # message format: team_id,team_password,plate_id
        message = f"{team_id},{team_password},0,{plate_id}"

        # Publish to the /license_plate topic
        self.license_plate_pub.publish(String(message))

    def stop_timer(self):
        '''
        @brief Stop the timer for the competition.

        This function sends a message on the /license_plate topic with license plate ID -1 (minus one) to stop the timer.

        @retval None
        '''

        # Team info
        team_id = "Gadalin"
        team_password = "bowser"
        plate_id = "PA55"

        # message format: team_id,team_password,plate_id
        message = f"{team_id},{team_password},-1,{plate_id}"

        # Publish to the /license_plate topic
        self.license_plate_pub.publish(String(message))
        
if __name__ == '__main__':
    rospy.init_node('robot_controller')
    rospy.set_param("/use_sim_time", True)

    rc = RobotController()
    # Initialize the controller
    delay = 4.0
    rc.initiate_controller(delay)
    # Start the control loop
    rc.start_timer()
    rc.control_loop()
    rospy.spin()
