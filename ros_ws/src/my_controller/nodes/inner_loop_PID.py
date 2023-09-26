#!/usr/bin/env python3

import rospy
# Import the topic message data objects
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# Import the line detection algorithm
import cv2
from cv_bridge import CvBridge
import vision_processing as vision

import threading

# Import the PID controller
# https://pypi.org/project/simple-pid/
from simple_pid import PID


## Class for processing the images and controlling the robot using a PID controller
#
class PIDAgent:
    """
    @brief Class constructor
    This method initializes the `PIDAgent` object by creating a publisher to velocity control
    and a subscriber object to the incoming raw image data.  It processes the images at the frame_out_interval.
    Each image updates the location of the center of the road and runs the error through PID.  
       
    
    @return None
    """
    def __init__(self):
        self.width = 1280

        # Flag to stop the controller - on/off switch from the robot controller
        self.continue_running = True

        ## PID controller for optimizing the control feedback
        #
        self.speed = .57  # Robot forward speed
        # self.pid = PID(1.4, .2, .03, setpoint=.35)  There are known good values
        self.pid = PID(2.2, .4, .3, setpoint=.3)  # Setpoint is biased to the right
        self.pid.sample_time = 0.02
        self.controlFeedback = 0
  
        # OpenCV to ROS Img message conversion
        self.bridge = CvBridge()
        
        # Set the update frequency for images/ PID
        self.last_pub_time = None
        self.frame_out_interval = 0.05
        
        #Store the most recent raw image
        self.image = None

        self.cmd_vel_pub = rospy.Publisher('R1/cmd_vel', Twist, queue_size=10)
        self.image_line_processing_pub = rospy.Publisher("R1/line_processing", Image, queue_size=10)
        self.image_sub = rospy.Subscriber("R1/pi_camera/image_raw", Image, self.image_callback)
        



    def image_callback(self, data):
        """
        @brief Callback function for image data
        This method is called whenever new image data is received on the "rrbot/camera1/image_raw" topic.
        It calls the `processImageToControlValue` method to process the image and determine the control feedback value,
        and creates a Twist message with linear x and angular z values to control the robot's movement.
        @param image_data (sensor_msgs/Image) : The raw image data received
        @return None
        """

        # Kill Switch
        if not self.continue_running:
            return

        # Process incoming image if it has been long enough since the last image was processed
        now = rospy.Time.now()
        if self.last_pub_time is None or (now - self.last_pub_time).to_sec() > self.frame_out_interval:
            # Convert the topic image to OpenCV format
            img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            # Convert the image to RGB color space (the color orderings are different between OpenCV and ROS)
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            # Run Line Recognition
            self.line_recognition()
            
            # Publish line recognition mask for debugging
            if(self.image_line_message_out is not None):
                self.image_line_processing_pub.publish(self.image_line_message_out)

            # Log the last time an image was published
            self.last_pub_time = rospy.Time.now()

    def line_recognition(self):
        """
        @brief Creates an hsv mask for the grey asphalt of the road in 353 competition, finds the centroid of the largest
               contour, then uses the x-postion to send inputs to PID turning/control output

        The mask for the road is handled a vision library function, hsv filtering that is specific to the road surface is used
        The mask is sent to a function that converts it to a control value, the control value is then fed into the PID control
        The output of the PID control determines the speed of turning on the constant forward speed robot.  Finally the line out
        message is updated with the mask image, it is automatically published in callback for debugging in rqt_viewer

        @modifies self.image_line_message_out
        @param self The object pointer
        @return None
        """

        mask = vision.lineVision(self.image)      
        self.processImageToControlValue( mask)        

        self.image_line_message_out = self.bridge.cv2_to_imgmsg(mask, "mono8")

    def processCentroid(self, image_data):
        """
        @brief Takes a threshholded binary mask and finds the centroid of the largest contour, returns -1 if none found

        A contour search is done at first to get a list of contours.  If there is no valid contour or the x-centroid has
        invalid math opreration, then a sentinel value of -1 is returned

        @param image_data - the binary mask to process
        @return centroid_x - the centroid of the largest contour shape
        """
        
        # TODO: Fix the issue with division by zero errors by implementing a guard
                # A -1 sentinel value can be used for invalid or non-existent contours
                # Return this if division by zero is going to happen for centroid calcs

        # Find contours in the mask
        contours, _ = cv2.findContours(image_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If there are no contours, return None
        if len(contours) == 0:
            centroid_x = -1
        else:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate the centroid of the largest contour
            M = cv2.moments(largest_contour)
            centroid_x = int(M['m10'] / M['m00'])

        #print("Centroid of the largest contour:", centroid)

        return centroid_x

    def processImageToControlValue(self, image_data):
        """
        @brief Takes a binary threshold image, runs it through PID and updates a motion control value

        Checks if a valid environment data input (centroid of line following) is given using -1 sentinel.  
        If no valid data is given, continue with the previous data storeed in self.pixelpoint
        The pixel column of the centroid is converted to a percentage of the screen width and we subtract
        1/2 to get error scaled from -.5 to .5 for offset from center of screen.  Multiply by 8 to rescale 
        from -4 to 4.  The error is run through PID to get a turn value, the contrl feedback is published to twist

        @param image_data - the binary mask to process
        @return None
        """

        centroid_x = self.processCentroid(image_data)
        
        # update pixelpoint of the line if it is defined (not -1)
        if (centroid_x > 0):
            self.pixelpoint = centroid_x
        
        # calculate the error between the setpoint and the current position
        error = 8*(self.pixelpoint/self.width - 1 / 2)

        self.controlFeedback = self.pid(error)   

        velocity_msg = Twist()
        velocity_msg.linear.x = self.speed
        velocity_msg.angular.z = self.controlFeedback
        self.cmd_vel_pub.publish(velocity_msg)    
        return

    def stop_agent(self):
        """
        @brief kill switch for agent

        Allows for an external script to terminate the linefollowing agent

        @return None
        """
        # Setting this to false terminates the image callback process    
        self.continue_running = False
        
        # Send the stop robot message to overide last motion message
        twist = Twist()
        twist.linear.x = 0
        twist.angular.z = 0
        self.cmd_vel_pub.publish(twist)

## Only run the node if the node is executed as a script
#
if __name__ == '__main__':
    ## Initialize the node and name it.
    #
    rospy.init_node('inner_loop_pid')

    pid_agent = PIDAgent()
   
    # Start a timer to stop the agent after 30 seconds 
    # TODO: This was explored as a way to terminate but stop_agent() also works now 
            # I think threading can be removed
    stop_timer = threading.Timer(30.0, pid_agent.stop_agent)
    stop_timer.start()

    rospy.spin()
