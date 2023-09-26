#!/usr/bin/env python3

import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from collections import defaultdict
from std_msgs.msg import String

import pathlib


from keras.models import model_from_json
import vision_processing as vision

class PlateDetection(object):
    '''
    @brief PlateDetection: A class for detecting license plates in a parking lot using computer vision and deep learning.

    This class is responsible for detecting license plates in a parking lot by subscribing to a camera feed and using
    Convolutional Neural Networks (CNNs) for image processing. It also handles the publishing of intermediate
    processing stages for analysis and debugging purposes.

    The class implements the following functionality:
    - Load and manage CNN models for parking ID and plate detection.
    - Subscribe to a camera feed and process images in real time.
    - Process images using HSV color space and character recognition.
    - Publish intermediate image processing stages for live analysis and debugging.
    - Store relevant image and processing information, such as homography samples and timestamps.

    @retval None
    '''

    def __init__(self):
        # CNN Models to Load
        self.parking_ID_model = None
        self.plate_model = None
        
        # Load the CNN models
        self.load_models()

        # Initialize the dictionary
        self.parking_data = defaultdict(lambda: defaultdict(int))

        self.bridge = CvBridge()

        # Publish to the license_plate topic to continuously update the parking data
        self.license_plate_pub = rospy.Publisher('license_plate', String)

        # Subscribe to the camera image and joystick topics (gives camera feed and direct access to controller)
        self.image_sub = rospy.Subscriber('R1/pi_camera/image_raw', Image, self.callback_image)
        # Create a subscriber for the kill command
        rospy.Subscriber('kill_command', String, self.kill_callback)
        self.kill_pub = rospy.Publisher('kill_command', String)


        # Three image topics are used so that many different stages of video processing or ideas can be tested live
        self.image_hsv_processing_pub = rospy.Publisher("R1/hsv_processing", Image)
        self.image_char_processing_pub = rospy.Publisher("R1/char_processing", Image)

        # Store image from subscribed topic
        self.image = None
        # Store the homography image
        self.warped = None
        # Store the homography sample for CNN training
        self.homography_sample = None

        # Store the last time an image was captured from image_raw
        self.last_capture_time = None
        # Store the last time an image was published
        self.last_pub_time = None

        # Time of last image published
        self.frame_out_interval = .001 # seconds

        # Storage for processed images
        self.image_hsv_message_out = None
        self.image_char_message_out = None

    def kill_callback(self, msg):
        '''
        @brief Listens for a kill command for the node, when called it will stop the competition timer

            When termination conditions are met for the competion halt all further plate readings by shutting 
            down the node after sending the stop timer special message (parkingID -1)            

        @retval None
        '''
        # If the message is 'kill', shut down the node
        if msg.data == 'kill':
            # Set your team ID and password
            team_id = "Gadalin"
            team_password = "bowser"

            # If the location is 8 and count is at least 4, publish the additional message
            plate_id = "PA55"
            message = f"{team_id},{team_password},-1,{plate_id}"
            self.license_plate_pub.publish(String(message))

            # Shutdown the node
            rospy.signal_shutdown('Kill command received')
    
    def load_models(self):
        '''
        @brief Load the CNN models for parking ID and plate detection from the specified directory.

        This method loads the pre-trained CNN models from JSON and H5 files in the given directory.
        The models are used for detecting parking spots and license plates from the camera feed.

        @retval None
        '''

        # Define the path to CNN directory
        PATH = pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve() / "cnn_trainer"

        # Define a function to load CNN models from JSON and H5 files that
        # Share the same name in the same PATH directory
        def load_model(model_name):
            # load json and create model
            with open(os.path.join(PATH, model_name + ".json"), 'r') as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            # Load the model weights
            model.load_weights(os.path.join(PATH, model_name + ".h5"))
            return model
        
        # Load the models
        self.parking_ID_model = load_model("cnn_parking_numbers/model2")
        self.plate_model = load_model("cnn_alpha/modelC4rebuild")

    def callback_image(self, data):
        '''
        @brief Callback function that processes the incoming image from the camera feed.

        This method is triggered when a new image is received from the camera feed. It checks if enough
        time has passed since the last image was processed, and if so, it proceeds to call the machine_vision
        method to process the image and then publishes the processed images.

        @param data (Image): The incoming image from the camera feed.

        @retval None
        '''


        # Process incoming image if it has been long enough since the last image was processed
        now = rospy.Time.now()
        if self.last_pub_time is None or (now - self.last_pub_time).to_sec() > self.frame_out_interval:
            self.machine_vision(data)  

            # Publish the processed images to all channels
            self.publishImages()
    
    def publishImages(self):
        '''
        @brief Publish the processed images to their respective topics.

        This method checks if the processed images (image_hsv_message_out and image_char_message_out) are not None,
        and if so, publishes them to their respective topics. It also logs the last time an image was published.  These
        are useful debugging panels to be viewed with rqt_viewer

        @retval None
        '''

        #Publish the processed images
        if(self.image_hsv_message_out is not None):
            self.image_hsv_processing_pub.publish(self.image_hsv_message_out)
        if(self.image_char_message_out is not None):
            self.image_char_processing_pub.publish(self.image_char_message_out)            
        # Log the last time an image was published
        self.last_pub_time = rospy.Time.now()

    def machine_vision(self, data):
        '''
        @brief Perform image processing and plate detection on the incoming image from the camera feed.

        This method processes the incoming image by converting it to OpenCV format and applying color space
        conversions, filtering, and contour detection. If a plate is found, it calls the processPlacard method to
        perform character recognition on the cropped image. It also stores the processed image in the
        image_hsv_message_out field for later publishing.

        @param data (Image): The incoming image from the camera feed.

        @retval None
        '''

        # Convert the topic image to OpenCV format
        img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        # Convert the image to RGB color space (the color orderings are different between OpenCV and ROS)
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hsv = vision.removeSky(hsv)

        # Get the white and blue masks by selective HSV filtering
        mask_white, mask_blue = vision.getWhiteBlueMasks(hsv)   

        # Filter out skinny lines from the white mask
        mask_white = vision.filterSkinnyLines(mask_white)
        # Combine the blue and white masks
        mask = cv2.bitwise_or(mask_blue, mask_white)


        # FILTERING FOR CONTOURS OF INTEREST        
        white_contours, blue_contours = vision.getContours(mask_white, mask_blue)
        # Create binary image
        thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        # Convert the thresh array to a color image so it can be published layered on raw image
        thresh_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Draw blue contours on thresh_image
        for contour in blue_contours:
            cv2.drawContours(thresh_image, [contour], -1, (255, 0, 0), 2)
        
        # Iterate white contours, draw onto thresh_image if they are a plate, return plate coords
        rect = self.findPlates(white_contours, mask_blue, thresh_image)

        full_height = vision.addSky(thresh_image, self.image)
        # Overlay the threshholding on top of the original image (gives a nice effect for machine vision)
        result = cv2.addWeighted(self.image, 0.18, full_height, 0.7, 0)

        # Convert the result to a ROS imgmsg and save it to be published
        try:
            self.image_hsv_message_out = self.bridge.cv2_to_imgmsg(result, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Call the CNN to identify the parking spot number if a plate was found
        if rect is not None:
            # If a plate is found, run character recognition on cropped image
            self.processPlacard(rect, vision.removeSky(self.image))

    def findPlates(self, white_contours, mask_blue, thresh_image):
    
        '''
        @brief Finds the coordinates of the plates in the image, filters by area and aspect ratio.

        This function looks for contours that are 4 sided polygons in the given binary image (thresh_image).
        It filters the contours by area and aspect ratio, and draws them on the thresh_image if they meet
        the criteria. Contours that do not meet the aspect ratio test are drawn in yellow, while those that
        pass the test are drawn in green.

        @param white_contours: A list of contours that are white.
        @param mask_blue: Location of blue pixels of parked cars.
        @param thresh_image: A binary image that has been thresholded.

        @returns A list of coordinates of the plates in the image. If no plates are found, returns None.

        @raises None
        '''

        PIXEL_TOLERANCE = 5 #Distance to search for blue car from white plate

        # Minimum aspect ratio threshold for filtering out skinny lines
        MIN_ASPECT_RATIO = .5
        MAX_ASPECT_RATIO = 1.0

        # Draw white contours on thresh_image and filter for placards
        for contour in white_contours:
            # Check if the contour is a rectangle by seeing if Polygon approximation has 4 corners
            approx = cv2.approxPolyDP(contour, 0.12*cv2.arcLength(contour, True), True)
            
            if len(approx) == 4:
                # Calculate and test aspect ratio of the rectangle
                rect = cv2.boundingRect(approx)
                aspect_ratio = min(rect[2], rect[3]) / max(rect[2], rect[3])
                
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

                    # Check that there is at least one blue pixel PIXEL_TOLERANCE pixels to the left of bottom left point or to right of right bottom
                    if (bottomleft[0] > PIXEL_TOLERANCE and  np.any(mask_blue[bottomleft[1], bottomleft[0]-PIXEL_TOLERANCE:bottomleft[0]]) )\
                        and (bottomright[0] < thresh_image.shape[1]-PIXEL_TOLERANCE and np.any(mask_blue[bottomright[1], bottomright[0]:bottomright[0]+PIXEL_TOLERANCE])):

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
    
    def getCorners(self, pts):
        '''
        @brief Returns the corner points of the plate in the image.

        This function takes a list of coordinates (rect) that define the plate in the image and returns
        the corner points of the plate sorted in order: topleft, topright, bottomright, bottomleft.

        @param rect: A list of coordinates of the plate in the image.

        @returns A list of corner points of the plate in the image sorted in order: topleft, topright, bottomright, bottomleft.

        @raises None
        '''


        # Find the two highest and two lowest points
        sorted_pts = pts[np.argsort(pts[:, 1])]
        top_pts = sorted_pts[:2]
        bottom_pts = sorted_pts[2:]

        # Identify leftmost and rightmost highest points
        topleft = top_pts[np.argmin(top_pts[:, 0])]
        topright = top_pts[np.argmax(top_pts[:, 0])]
        bottomleft = bottom_pts[np.argmin(bottom_pts[:, 0])]
        bottomright = bottom_pts[np.argmax(bottom_pts[:, 0])]

        # Order the points as top-left, top-right, bottom-right, bottom-left
        rect = np.array([topleft, topright, bottomright, bottomleft], dtype=np.int32)

        return rect

    def processPlacard(self, rect, img):
        """
        @brief  Perform character recognition on the detected plate and draw results on the image.

        This method processes the detected plate by applying homography, slicing out the areas of interest,
        and processing the images through the CNN models. The results are drawn on the warped image for display
        and debugging, which can be visualized using the rospy.Publisher("R1/char_processing", Image) topic.
        It also updates the image_char_message_out field with the processed image for later publishing.

        @param rect (list): The coordinates of the detected plate.
               img (numpy.ndarray): The input image without sky.
        """

        self.warped = vision.getWarpedImage(rect, img)
        # save a homgraphy sample that will not be drawn on for saving to file for the CNN
        self.homography_sample = self.warped.copy()

        #Slice out all 5 areas of interest starting with
        parking_id, sliced_images = vision.getSlicedImages(self.warped)

        # Process the images through the CNN
        y_predict = self.processParkingIDCNN(parking_id)
        char_predict, max_vals = self.processCharactersCNN(sliced_images)

        # Draw the results on the warped image for display and debugging
        self.warped = vision.drawParkingIDOverlay(y_predict, self.warped)
        self.warped = vision.drawCharactersOverlay(char_predict, max_vals, self.warped)

        # Apply any forced character corrections
        char_predict = self.correct_characters(char_predict)

        # Add result to dictionary
        pred_label = np.argmax(y_predict)+1
        plate = str(char_predict)
        location = int(pred_label)
        self.add_plate_reading(plate, location)
        #Send updated data to the server
        self.publish_plate_reading(location)

        self.display_most_common_plates()

        # Convert the result to a ROS imgmsg and update image_char_message_out
        try:
            self.image_char_message_out = self.bridge.cv2_to_imgmsg(self.warped, "bgr8")
        except CvBridgeError as e:
            print(e)

    def processParkingIDCNN(self, parking_id_img):
        '''
        @brief Perform character recognition on the sliced image of a parking ID

        @param parking_id_img (np.array()): The image of a parking ID

        @retval y_predict (double vector) : The prediction for the parking ID from the CNN model
        '''

        # Convert the image to grayscale
        parking_id_img  = cv2.cvtColor(parking_id_img , cv2.COLOR_BGR2GRAY)
        # Resize the image to 20x25
        parking_id_img = cv2.resize(parking_id_img , (25, 20))
        # Set as np array 
        parking_id_img = np.array(parking_id_img)
        # Normalize the image for CNN input (from 0 to 1)
        parking_id_img = parking_id_img/255
        # Predict the parking ID
        y_predict = self.parking_ID_model.predict(parking_id_img.reshape(1, 20, 25, 1) , verbose=0)

        return y_predict

    def processCharactersCNN(self, img_set):
        '''
        @brief Process the character images through the CNN model.

        This method preprocesses the character images, generates position vectors, and feeds them to the CNN
        model for prediction. It returns the predicted characters and their maximum values.

        @param img_set (list): The list of character images to process. The list should be in the order left to right.

        @returns A tuple containing two lists:
                    - char_vals (list): The predicted characters.
                    - max_values (list): The maximum values of the predictions.

        @raises None
        '''

        labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        # Preprocess the images and generate postion vectors to feed to the CNN
        pos_vectors = []
        for i in range(len(img_set)):
            img_set[i] = vision.preprocessCharacterImages(img_set[i])
            one_hot = np.zeros(4)
            one_hot[i] = 1
            pos_vectors.append(one_hot)        

        # Reshape to the CNN inputs
        processed_images = np.reshape(img_set, (-1, 40, 40, 1))
        pos_vectors = np.array(pos_vectors)

        # Make predictions on all 4 characters
        predictions = self.plate_model.predict([processed_images, pos_vectors], verbose=1)

        # Initialize empty lists to store the predicted characters and maximum values
        char_vals = []
        max_values = []
        # Iterate over each prediction and get the corresponding predicted character and maximum value
        for p in predictions:
            # Take the highest value as the prediction
            predicted_label = np.argmax(p)
            # Get the maximum value of the prediction
            max_value = np.amax(p)
            # Convert the predicted label to character
            char_val = labels[predicted_label]
            # Append the character and maximum value to the corresponding lists
            char_vals.append(char_val)
            max_values.append(max_value)

        return char_vals, max_values
    
    def correct_characters(self, char_predict):
        '''
        @brief Apply a basic character correction override for characters that are misread

        Corrects erronous plate readings and prints an update to console

        @param char_predict (str): The plate string, e.g. 'AA56'.

        @retval corrected_chars (str): The corrected value for the plate reading
        '''
        swaps = [('8', 'B'), ('0','O') , ('5', 'S'), ('1', 'I'), ('2', 'Z')]
        corrected_chars = char_predict.copy()

        def swap_chars(pairs):
            for i in range(2):
                for pair in pairs:
                    if corrected_chars[i] == pair[0]:
                        print(f"ERROR CORRECTION {corrected_chars[i]}->{pair[1]}")
                        corrected_chars[i] = pair[1]

            for i in range(2, 4):
                for pair in pairs:
                    if corrected_chars[i] == pair[1]:
                        print(f"ERROR CORRECTION {corrected_chars[i]}->{pair[0]}")
                        corrected_chars[i] = pair[0]

        swap_chars(swaps)

        return corrected_chars

    def add_plate_reading(self, plate, location):
        '''
        @brief Add a plate reading to the parking_data dictionary.

        This function adds a plate reading to the parking_data dictionary which holds all of the plate readings made so far.
        The plate reading consists of a plate string (plate) and a parking location ID (location). The count for an existing
        read is updated by one.

        @param plate (str): The plate string, e.g. 'AA56'.
        @param location (int): The parking location ID, e.g. 4.

        @retval None
        '''

        self.parking_data[location][plate] += 1

    def get_most_common_plate(self, location):
        '''
        @brief Get the most common plate at a specific location.

        This function retrieves the most common plate at a specific parking location by counting the occurrences of
        each plate string in the parking_data dictionary. If no plates are found at the location, it returns 'N/A' as
        the plate string and 0 as the count.

        @param location (int): The parking location ID.

        @returns A tuple containing:
                    - str: The most common plate or 'N/A' if no plates found.
                    - int: The count of the most common plate or 0 if no plates found.

        @raises None
        '''

        plates_counts = self.parking_data[location]
        if not plates_counts:
            return "N/A", 0

        most_common_plate, count = max(plates_counts.items(), key=lambda x: x[1])

        return most_common_plate, count

    def display_most_common_plates(self):
        '''
        @brief Display the most common plate and count for all locations (1 to 8) in rqt_console.

        This function retrieves the most common plate and count for each parking location (1 to 8) by calling the
        get_most_common_plate function for each location. It then displays the results in the rqt_console.

        @retval None
        '''

        log_messages = []

        for location in range(1, 9):
            most_common_plate, count = self.get_most_common_plate(location)
            log_message = f"Location {location}: Most common plate is {most_common_plate} with a count of {count}"
            log_messages.append(log_message)

        # Concatenate all the log messages into a single string
        log_message = '\n'.join(log_messages)

        # Send an info log message with all the locations and counts
        print('\n' + log_message)
    
    def publish_plate_reading(self, location):
        '''
        @brief Publish the most common plate and count for a specific location.

        This function retrieves the most common plate and count for a specific parking location by calling the
        get_most_common_plate function. It then publishes the results to the license_plate topic.

        There is a special case where if we have reached a threshold count on plate 8, the competition clock will be stopped
        The stoppage is communicated to both the controller and plate reader via the kill Publisher topic

        @param location (int): The parking location ID.

        @retval None
        '''

        # Get the most common plate at the location
        most_common_plate, count = self.get_most_common_plate(location)

        # Extract the first four alphanumeric characters from the list
        alphanumeric_chars = [char for char in most_common_plate if char.isalnum()]
        most_common_plate_str = ''.join(alphanumeric_chars[:4])

        # Set your team ID and password
        team_id = "Gadalin"
        team_password = "bowser"

        # Create the message string in the required format
        message = f"{team_id},{team_password},{location},{most_common_plate_str}"

        # Publish the message to the license_plate topic
        self.license_plate_pub.publish(String(message))

        # Detect an early finish and stop timer
        if location == 8 and count >= 4:
            self.kill_pub.publish('kill')


if __name__ == '__main__':
    rospy.init_node('plate_detection')
    PlateDetection()
    rospy.spin()
