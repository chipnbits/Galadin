#!/usr/bin/env python3

import rospy

import os
import datetime
import csv
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist  # Import the Twist message type
import numpy as np


class PlateDataImages:
    def __init__(self, nrobots):
        self.SKY_CUTOFF = 340  # The rejection line for the sky
        # Track how many cameras are postioned for data mining
        self.nrobots = nrobots
        rospy.loginfo("nrobots: {}".format(self.nrobots))
        self.bridge = CvBridge()

        # Store all the data subscribers
        self.image_subs = []
        self.image_count = 0

         # Create a directory to save images captured with the controller button
        self.output_dir_raw = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved_images', 'raw'))
        self.output_dir_warped = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved_images', 'warped'))
        os.makedirs(self.output_dir_raw, exist_ok=True)  # Explicitly create output directory if it doesn't exist
        os.makedirs(self.output_dir_warped, exist_ok=True)  # Explicitly create output directory if it doesn't exist

        rospy.sleep(1)  # delay for plate gen before capturing images
        # Gather plate labels from the csv file
        self.prefixes = []
        self.setprefixes()
                # Initialize a list of publishers, one for each robot
        self.twist_pubs = []
        rospy.sleep(2)
        # for i in range(1, nrobots+1):
        #     topic = "/P{}/cmd_vel".format(i)
        #     rospy.loginfo("Publishing twist commands on topic: {}".format(topic))

        #     # Create a publisher with a buffer size of 10 messages
        #     twist_pub = rospy.Publisher(topic, Twist, queue_size=10)
        #     self.twist_pubs.append(twist_pub)

        # # Publish twist commands for each robot to turn left or right for a random duration at the start of the for loop
        # for i in range(1, nrobots+1):
        #     # Choose a random direction and duration for the twist command
        #     direction = 1 if np.random.rand() < 0.5 else -1
        #     duration = np.random.uniform(0.0, .1)

        #     # Create the Twist message with the random angular velocity
        #     twist_msg = Twist()
        #     twist_msg.angular.z = direction * np.random.uniform(0.5, 1.0)
        #     self.twist_pubs[i-1].publish(twist_msg)

        #     rospy.sleep(duration)

        #     self.twist_pubs[i-1].publish(twist_msg)

        #     # Publish stop Twist message after sleep
        #     twist_stop = Twist()
        #     twist_stop.angular.z = 0.0
        #     self.twist_pubs[i-1].publish(twist_stop)
        #     rospy.sleep(.3)         
            


        for i in range(1, nrobots+1):
            topic = "/P{}/pi_camera/image_raw".format(i)
            rospy.loginfo("Subscribed to topic: {}".format(topic))

            # Wait for an image message to be published on the topic
            msg = rospy.wait_for_message(topic, Image)

            # Convert the message to a cv2 image and save it
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(e)
                return
            plate_label = self.prefixes[i-1] # Get the prefix for this robot

            # One night of raw images was about 10GB, so we're not saving them anymore
            # filename = f'{self.output_dir_raw}/P{i}_{plate_label}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            # cv2.imwrite(filename, cv_image)
            # rospy.loginfo(f"Saved {filename}")

            raw_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            warped = self.machine_vision(raw_image)
            if warped is not None:
                warped = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
                filename = f'{self.output_dir_warped}/P{i}_{plate_label}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                cv2.imwrite(filename, warped)
                rospy.loginfo(f"Saved {filename}")
            self.image_count += 1

    # Read the prefixes from the csv file
    def setprefixes(self):

        # Get the path to the plates.csv file
        csv_path = os.path.expanduser('~/ros_ws/src/2022_competition/enph353/enph353_gazebo/scripts/plates.csv')

        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < 8:
                    self.prefixes.append(row[0])
        rospy.loginfo("Prefixes: {}".format(self.prefixes)) 

    def machine_vision(self, img):
        """
        Uses computer vision techniques to identify license plates in the incoming image and runs
        character recognition algorithms to identify the characters on the plates.

            - Converts the incoming image to HSV color space.
            - Applies several filters and thresholds to the incoming image in order to identify regions
                that are likely to contain license plates.
            - Further analyzes these regions to verify that they actually contain license plates.
            - Runs a final check that the plate is bordered by a blue car on both sides to prevent partial plates
                from being identified.
            - Runs character recognition algorithms on these regions to determine the characters on the plates.
        
        The following instance variables of `self` are modified within this method:
        
            - self.image_hsv_message_out: the processed image in HSV format, with contours drawn around license 
                plates that have been identified. Contours around cars.
            - self.image_char_message_out: There is a function call to process image if a plate is found

        @param self: An instance of the `MachineVision` class.
        """
        # Define minimum area threshold for white contours
        MIN_WHITE_AREA = 4000
        MIN_BLUE_AREA = 10000

        # Convert the incoming image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Cutout the bottom half of the image to remove useless information (sky)
        height, width = hsv.shape[:2]
        hsv = hsv[self.SKY_CUTOFF:height, 0:width]
        
        # Get the white and blue masks by selective HSV filtering
        mask_white, mask_blue = self.getWhiteBlueMasks(hsv)   
        # Filter out skinny lines from the white mask
        mask_white = self.filterSkinnyLines(mask_white)     
        # END OF MASKING PROCESS

        # Combine the blue and white masks
        mask = cv2.bitwise_or(mask_blue, mask_white)
        # FILTERING FOR CONTOURS OF INTEREST        
        # Create binary image
        thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        # Convert the thresh array to a color image
        thresh_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Find blue white contours
        white_contours, hierarchy = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, hierarchy = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter out small white contours
        white_contours = [contour for contour in white_contours if cv2.contourArea(contour) > MIN_WHITE_AREA]
        # Filter out skinny white contours
        white_contours = [contour for contour in white_contours if (cv2.boundingRect(contour)[2]/cv2.boundingRect(contour)[3]) < 5]
        # Filter out small blue contours
        blue_contours = [contour for contour in blue_contours if cv2.contourArea(contour) > MIN_BLUE_AREA]

        # Draw blue contours on thresh_image
        for contour in blue_contours:
            cv2.drawContours(thresh_image, [contour], -1, (255, 0, 0), 2)


        # Iterate white contours, draw onto thresh_image if they are a plate, return plate coords
        rect = self.findPlates(white_contours, mask_blue, thresh_image)

        if rect is not None:
            # If a plate is found, run character recognition
            return self.char_search(rect, img[self.SKY_CUTOFF:height, 0:width])
        else: 
            return None
            


        # # Adjust thresh image to be the same size as the original image
        # extended_height = np.zeros_like(self.image)
        # extended_height[self.SKY_CUTOFF:, :] = thresh_image
        # # Overlay the threshholding on top of the original image (gives a nice effect for machine vision)
        # result = cv2.addWeighted(self.image, 0.18, extended_height, 0.7, 0)

        # # Convert the result to a ROS imgmsg
        # try:
        #     self.image_hsv_message_out = self.bridge.cv2_to_imgmsg(result, "bgr8")
        # except CvBridgeError as e:
        #     print(e)
        

    def getWhiteBlueMasks(self, img):
        """
        Returns a mask for the white and blue regions of the image.

        Args:
            img: The image to be processed.

        Returns:
            A tuple containing the masks for the white and blue regions of the image.

        Raises:
            None
        """

        # Define the RGB values for the blue color of parked cars
        rgb_blue = (50, 50, 150)

        # Define the RGB value for the white color of the plates
        rgb_white = (199, 199, 199)

        # Convert the RGB values to HSV color space
        hsv_blue = cv2.cvtColor(np.uint8([[rgb_blue]]), cv2.COLOR_RGB2HSV)[0][0]
        hsv_white = cv2.cvtColor(np.uint8([[rgb_white]]), cv2.COLOR_RGB2HSV)[0][0]

        # Argument order is Hue, Saturation, Value
        # Define the lower and upper bounds for the blue color
        hsv_lower_blue = np.array([hsv_blue[0] - 15, 50, 50])
        hsv_upper_blue = np.array([hsv_blue[0] + 15, 255, 255])

        # Argument order is Hue, Saturation, Value
        # Define the lower and upper bounds for the white color (tight threshold)
        hsv_lower_white = np.array([hsv_white[0] - 5, 0, 95])
        hsv_upper_white = np.array([hsv_white[0] + 5, 20, 240])

        # Create a mask for the blue color
        mask_blue = cv2.inRange(img, hsv_lower_blue, hsv_upper_blue)

        # Create a mask for the white color
        mask_white = cv2.inRange(img, hsv_lower_white, hsv_upper_white)

        return mask_white, mask_blue

    def filterSkinnyLines(self, mask):
        # Apply morphological opening to the white mask to remove skinny lines
        # Create structuring element for vertical line
        line_length = 10
        line_width = 2
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, line_length))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, line_kernel)

        # Apply closing operation to fill in gaps and smooth out boundaries
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
        return mask  

    def findPlates(self, white_contours, mask_blue, thresh_image):
        """
        Finds the coordinates of the plates in the image, filters by area and aspect ratio
        Looks for contours that are 4 sided polygons.  Draws the contours on the thresh_image that match
        If they don't meet aspect ratio test, yellow, if they past test, the countour is green

        Args:
            white_contours: A list of contours that are white
            mask_blue: location of blue pixels of parked cars
            thresh_image: A binary image that has been thresholded

        Returns:
            A list of coordinates of the plates in the image

        Raises:
            None
        """
        PIXEL_TOLERANCE = 5 #Distance to search for blue car from white plate

        # Minimum aspect ratio threshold for filtering out skinny lines
        MIN_ASPECT_RATIO = .5
        MAX_ASPECT_RATIO = 1.0

        # Draw white contours on thresh_image and filter for placards
        for contour in white_contours:
            # Check if the contour is a rectangle
            approx = cv2.approxPolyDP(contour, 0.12*cv2.arcLength(contour, True), True)
            
            if len(approx) == 4:
                # Calculate aspect ratio of the rectangle
                rect = cv2.boundingRect(approx)
                aspect_ratio = min(rect[2], rect[3]) / max(rect[2], rect[3])
                # print(aspect_ratio)

                
                if aspect_ratio > MIN_ASPECT_RATIO and aspect_ratio < MAX_ASPECT_RATIO:
                    # Extract the corner points
                    pts = approx.reshape(4,2)

                    # Find the two highest and two lowest points
                    sorted_pts = pts[np.argsort(pts[:, 1])]
                    top_pts = sorted_pts[:2]
                    bottom_pts = sorted_pts[2:]

                    # Identify leftmost and rightmost highest points
                    topleft = top_pts[np.argmin(top_pts[:, 0])]
                    topright = top_pts[np.argmax(top_pts[:, 0])]
                    bottomleft = bottom_pts[np.argmin(bottom_pts[:, 0])]
                    bottomright = bottom_pts[np.argmax(bottom_pts[:, 0])]

                   # Check if bottom left or bottom right is near the edge of the thresh_image
                    if (bottomleft[1] > PIXEL_TOLERANCE) or (bottomright[1] < (thresh_image.shape[1] - PIXEL_TOLERANCE)):

                        # Extend the bottom corners by 30% of the height of each side
                        height_left = bottomleft[1] - topleft[1]
                        height_right = bottomright[1] - topright[1]
                        bottom_left = bottomleft + np.array([0, 0.27*height_left])
                        bottom_right = bottomright + np.array([0, 0.27*height_right])

                        # Order the points as top-left, top-right, bottom-right, bottom-left
                        rect = np.array([topleft, topright, bottom_right, bottom_left], dtype=np.float32)

                        # Draw rectangle
                        cv2.drawContours(thresh_image, [rect.astype(np.int32)], -1, (0, 255, 0), 2)

                        # We have a plate match!
                        return rect                      
                            
                    else:
                        cv2.drawContours(thresh_image, [contour], -1, (0, 255, 255), 2)

        return None
                    
    def char_search(self, rect, img):
        """
        This function performs character recognition on a region of interest (ROI) defined by a rectangle in the input image.
        Time: Takes less than 1ms as of March 3rd

        - Calculates the width and height of the target warped image based on the corner points of the rectangle.
        - Defines the target points for the perspective transformation and applies it to the input image.
        - Checks the aspect ratio of the warped image to ensure that it is within a certain range.
        - Splits the warped image into top and bottom parts.
        - Thresholds the top and bottom parts for black and blue respectively.
        - Defines the RGB values for the blue color and converts them to HSV color space.
        - Defines the lower and upper bounds for the blue color and applies a mask to the bottom part of the warped image.
        - Converts the bottom part of the warped image to grayscale and splits it into its channels.
        - Combines the top and bottom parts and draws a red line at the slice point.
        - Converts the result to a ROS imgmsg and updates `image_char_message_out`.
        - Runs character recognition algorithms on the warped image to determine the characters on the license plate.

        @param self: An instance of this class.
        @param rect: A list of four corner points that define the ROI rectangle in the input image.
        """

# Define a fixed output size for the warped image
        OUTPUT_SIZE = (200, 240)

        # Extract the corner points
        topleft, topright, bottom_right, bottom_left = rect

        # Determine the width and height of the rectangle to warp to
        width = max(np.linalg.norm(topleft - topright), np.linalg.norm(bottom_left - bottom_right))
        height = max(np.linalg.norm(topleft - bottom_left), np.linalg.norm(topright - bottom_right))

        # Calculate the scaling factors for the width and height
        scale_x = OUTPUT_SIZE[0] / width
        scale_y = OUTPUT_SIZE[1] / height

        # Scale up the width and height by the scaling factors
        width *= scale_x
        height *= scale_y

        # Define the target points for the perspective transformation
        target = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

        # Get the perspective transform matrix and warp the image
        M = cv2.getPerspectiveTransform(rect, target)
        warped = cv2.warpPerspective(img, M, (int(width), int(height)))
        return warped







if __name__ == '__main__':
    rospy.init_node('plate_data_images')
    nrobots = rospy.get_param('~nrobots')
    pdi = PlateDataImages(nrobots)
    rospy.spin()


