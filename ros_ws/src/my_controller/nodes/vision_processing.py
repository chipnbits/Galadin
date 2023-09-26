#!/usr/bin/env python3

import numpy as np
import cv2
import math

# Define constants for image processing
SKY_CUTOFF = 300  # The rejection line for the sky
SKY_SCANNER_CUTOFF = 400  # The rejection line for the sky for the sky scanner

## @defgroup Group1 Plate Processing
#  @brief Vision processing for plate processing

## @defgroup Group2 Pedestrian Vision
#  @brief Vision processing for pedestrians

## @defgroup Group3 Fixate Vision
#  @brief Vision processing for fixation on end of road


## @ingroup Group1
def removeSky(image):
    '''
    @brief Remove the sky portion of an image.

    This function removes the sky portion of an image by cutting out the top part of the image, reducing
    unnecessary information for processing.

    @param image (numpy.ndarray): The input image.

    @returns The resulting image with the sky portion removed (numpy.ndarray).

    @raises None
    '''

    # Cutout the bottom half of the image to remove useless information (sky)
    height, width = image.shape[:2]
    return image[SKY_CUTOFF:height, 0:width]

## @ingroup Group1
def addSky(thresh, img):
    '''
    @brief Add the sky portion back to an image.

    This function adds the sky portion back to an image by extending the processed image to match the
    original size. This is useful for converting between data processing size and the original size.

    @param thresh (numpy.ndarray): The thresholded image without the sky portion.
    @param img (numpy.ndarray): The original image.

    @returns The resulting image with the sky portion added (numpy.ndarray).

    @raises None
    '''

     # Adjust thresh image to be the same size as the original image
    extended_height = np.zeros_like(img)
    extended_height[SKY_CUTOFF:, :] = thresh
    return extended_height

## @ingroup Group1
def getWhiteBlueMasks(img):
    """
    @brief Returns masks for the white and blue regions of the input image.

    @param img (numpy.ndarray): The image to be processed.

    @return (tuple): A tuple containing the masks for the white and blue regions of the image.

    @throws None
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

## @ingroup Group1
def filterSkinnyLines(img):
    '''
    @brief Filter skinny white lines from an image.

    This function applies morphological operations to the input image to filter out skinny white lines,
    enhancing the plate detection process.

    @param img (numpy.ndarray): The input image.

    @returns The resulting image with skinny white lines filtered out (numpy.ndarray).

    @raises None
    '''

    # Apply morphological opening to the white mask to remove skinny lines
    # Create structuring element for vertical line
    line_length = 6
    line_width = 2
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, line_length))

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, line_kernel)

    # Apply closing operation to fill in gaps and smooth out boundaries
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, closing_kernel)
    return img  

## @ingroup Group1
def getContours(white, blue):
    """
    Get filtered white and blue contours from input masks.

    This function finds and filters the white and blue contours based on area and aspect ratio,
    enhancing the plate detection process.

    Args:
        white (numpy.ndarray): The white mask image.
        blue (numpy.ndarray): The blue mask image.

    Returns:
        tuple: A tuple containing two lists of contours: (white_contours, blue_contours)
    """
    # Define minimum area threshold for contours
    MIN_WHITE_AREA = 2500
    MIN_BLUE_AREA = 10000
     # Find blue white contours
    white_contours, hierarchy = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, hierarchy = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter out small white contours
    white_contours = [contour for contour in white_contours if cv2.contourArea(contour) > MIN_WHITE_AREA]
    # Filter out skinny white contours
    white_contours = [contour for contour in white_contours if (cv2.boundingRect(contour)[2]/cv2.boundingRect(contour)[3]) < 5]
    # Filter out small blue contours
    blue_contours = [contour for contour in blue_contours if cv2.contourArea(contour) > MIN_BLUE_AREA]   
     
    return white_contours, blue_contours

## @ingroup Group1
def getWarpedImage(rect, img):
    """
    Perform a perspective transformation on the input image using the provided rectangle.

    This function takes the corner points of a rectangle and warps the input image to a
    standard size, which can then be processed by the CNN.

    Args:
        rect (numpy.ndarray): A 2D array containing the corner points of the rectangle.
        img (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The warped image.
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

## @ingroup Group1
def getSlicedImages(warped):
    '''
    @brief Extract the individual characters and the parking ID from the warped image.

    This function calculates the positions of horizontal and vertical lines based on the size of the image
    and crops out the parking ID and the 4 plate characters, returning them in a tuple.

    @param warped (numpy.ndarray): The warped image.

    @returns A tuple containing:
                - numpy.ndarray: The parking ID image.
                - list: A list of cropped character images.

    @raises None
    '''
    # Calculate the positions of the horizontal and vertical lines based on the size of the image
    height, width, _ = warped.shape
    x1 = int(0.52 * width)
    x2 = int(0.97 * width)
    y1 = int(0.4 * height)
    y2 = int(0.72 * height)

    # Crop the rectangle formed by the four points
    parking_id_img = warped[y1:y2, x1:x2]

    x1 = int(0.05 * width)
    x2 = int(0.95 * width)
    y1 = int(0.8 * height)
    y2 = int(0.97 * height)

    x1_left = x1
    x2_left = int(x1 + (x2 - x1) / 2.3)
    mid_left = int((x2_left+x1_left)/2)

    x1_right = int(x2 - (x2 - x1) / 2.3)
    x2_right = x2
    mid_right = int((x2_right+x1_right)/2)
    
    # Crop the four rectangles with character position indexed
    cropped_images = [warped[y1:y2, x1_left:mid_left],
                    warped[y1:y2, mid_left:x2_left],
                    warped[y1:y2, x1_right:mid_right],
                    warped[y1:y2, mid_right:x2_right]]
    
    return parking_id_img, cropped_images

## @ingroup Group1
def preprocessCharacterImages(img):
    '''
    @brief Preprocess each of the plate characters for better recognition.

    This function resizes the input image, converts it to HSV color space, defines the RGB values for the
    blue color of parked cars, creates a mask for the blue color, and applies the mask to the blue channel
    of the original image. The resulting image is then converted to grayscale for better recognition.

    @param img (numpy.ndarray): The input character image.

    @returns The preprocessed character image in grayscale (numpy.ndarray).

    @raises None
    '''

    img = cv2.resize(img, (40, 40))
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the RGB values for the blue color of parked cars
    rgb_blue = (6, 6, 99)

    # Convert the RGB values to HSV color space
    hsv_blue = cv2.cvtColor(np.uint8([[rgb_blue]]), cv2.COLOR_RGB2HSV)[0][0]

    # Argument order is Hue, Saturation, Value
    # Define the lower and upper bounds for the blue color
    hsv_lower_blue = np.array([hsv_blue[0] - 1, 80, 50])
    hsv_upper_blue = np.array([hsv_blue[0] + 1, 255, 200])

    # Create a mask for the blue color
    mask = cv2.inRange(hsv, hsv_lower_blue, hsv_upper_blue)

    # Apply the mask to the blue channel of the original image
    blue_channel = img[:, :, 2]
    blue_parts = cv2.bitwise_and(blue_channel, blue_channel, mask=mask)

    # Convert the resulting image to greyscale
    grey_blue_parts = cv2.cvtColor(cv2.merge((blue_parts, blue_parts, blue_parts)), cv2.COLOR_RGB2GRAY)

    return grey_blue_parts

## @ingroup Group1
def drawParkingIDOverlay(y_predict, warped):
    '''
    @brief Draw a red overlay box around the parking ID area and display the predicted parking ID.

    This function draws a red overlay box around the parking ID area and displays the predicted parking ID.
    The input image is modified in place and the resulting image is returned.

    @param y_predict (numpy.ndarray): The predicted parking ID in one-hot encoding format.
    @param warped (numpy.ndarray): The input image.

    @returns The image with the parking ID overlay (numpy.ndarray).

    @raises None
    '''

# Check the aspect ratio of the warped image
    warped_height, warped_width, _ = warped.shape

    # Draw a red box onto the image
    line_color = (0, 0, 255)  # red color in BGR format
    line_thickness = 1  # thickness of the lines in pixels

    # Calculate the positions of the horizontal and vertical lines
    x1 = int(0.52 * warped_width)
    x2 = int(0.97 * warped_width)
    y1 = int(0.4 * warped_height)
    y2 = int(0.72 * warped_height)

    # Draw the horizontal lines
    cv2.line(warped, (x1, y1), (x2, y1), line_color, thickness=line_thickness)
    cv2.line(warped, (x1, y2), (x2, y2), line_color, thickness=line_thickness)

    # Draw the vertical lines
    cv2.line(warped, (x1, y1), (x1, y2), line_color, thickness=line_thickness)
    cv2.line(warped, (x2, y1), (x2, y2), line_color, thickness=line_thickness)

    # Overlay the predicted parking ID onto the image
    # Convert 1-hot vector to scalar label
    pred_label = np.argmax(y_predict)+1

    # Draw the predicted label on the image
    cv2.putText(warped, f"Parking ID: {pred_label}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return warped

## @ingroup Group1
def drawCharactersOverlay(char_predict, max_vals, warped):
    '''
    @brief Draw a blue overlay box around the plate characters area and display the predicted characters.

    This function calculates the positions of the horizontal and vertical lines, draws blue boxes around
    the plate characters area, overlays the predicted characters, and displays the list of max_vals at
    the top of the image. The input image is modified in place and the resulting image is returned.

    @param char_predict (list): The predicted characters.
    @param max_vals (list): The list of maximum values associated with the predicted characters.
    @param warped (numpy.ndarray): The input image.

    @returns The image with the characters overlay (numpy.ndarray).

    @raises None
    '''

    # Check the aspect ratio of the warped image
    warped_height, warped_width, _ = warped.shape

    # Draw a blue box onto the image
    line_color = (255, 0, 0)  # blue color in BGR format
    line_thickness = 1  # thickness of the lines in pixels 

    # Calculate the positions of the horizontal and vertical lines
    x1 = int(0.05 * warped_width)
    x2 = int(0.95 * warped_width)
    y1 = int(0.8 * warped_height)
    y2 = int(0.97 * warped_height)

    # Calculate the positions of the new vertical lines for the left and right rectangles
    x1_left = x1
    x2_left = int(x1 + (x2 - x1) / 2.3)
    x1_right = int(x2 - (x2 - x1) / 2.3)
    x2_right = x2

    # Draw the horizontal lines for the left rectangle
    cv2.line(warped, (x1_left, y1), (x2_left, y1), line_color, thickness=line_thickness)
    cv2.line(warped, (x1_left, y2), (x2_left, y2), line_color, thickness=line_thickness)

    # Draw the horizontal lines for the right rectangle
    cv2.line(warped, (x1_right, y1), (x2_right, y1), line_color, thickness=line_thickness)
    cv2.line(warped, (x1_right, y2), (x2_right, y2), line_color, thickness=line_thickness)

    # Draw the vertical lines for the left rectangle
    cv2.line(warped, (x1_left, y1), (x1_left, y2), line_color, thickness=line_thickness)
    cv2.line(warped, (x2_left, y1), (x2_left, y2), line_color, thickness=line_thickness)
    # Divider Line
    mid = int((x2_left+x1_left)/2)
    cv2.line(warped, (mid, y1), (mid, y2), line_color, thickness=line_thickness)

    # Draw the vertical lines for the right rectangle
    cv2.line(warped, (x1_right, y1), (x1_right, y2), line_color, thickness=line_thickness)
    cv2.line(warped, (x2_right, y1), (x2_right, y2), line_color, thickness=line_thickness)
    # Divider Line
    mid = int((x2_right+x1_right)/2)
    cv2.line(warped, (mid, y1), (mid, y2), line_color, thickness=line_thickness)

    # Define the font and font scale for the characters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

    # Draw the characters over the left rectangle
    cv2.putText(warped, char_predict[0], (x1_left + 5, y1 - 5), font, font_scale, line_color, thickness=2)
    cv2.putText(warped, char_predict[1], (x2_left - 5 - 30, y1 - 5), font, font_scale, line_color, thickness=2)

    # Draw the characters over the right rectangle
    cv2.putText(warped, char_predict[2], (x1_right + 5, y1 - 5), font, font_scale, line_color, thickness=2)
    cv2.putText(warped, char_predict[3], (x2_right - 5 - 30, y1 - 5), font, font_scale, line_color, thickness=2)

    # Draw the list of max_vals at the top of the image
    max_vals_text = 'Max Vals: {} {} {} {}'.format(*max_vals)
    max_vals_position = (int(0.05 * warped_width), int(0.2 * warped_height))  
    cv2.putText(warped, max_vals_text, max_vals_position, font, font_scale, line_color, line_thickness)
    

    return warped

## @ingroup Group2
def filterPedPants(img):
    """
    @brief Filters the input image for the pants of pedestrians.
    
    Takes an input image and returns a mask that highlights the pants of pedestrians 
    A tuned hsv filter is used to filter out the pants of pedestrians. The image is enhanced 
    with dilation to give a large white blob for the pants of pedestrians that can be tracked

    @param img (numpy.ndarray): The input image to be processed.

    @returns mask_blue_processed (numpy.ndarray): A mask highlighting the pants of pedestrians.

    @raise None
    """

    # Define the RGB values for the blue color of parked cars
    rgb_blue = (51, 86, 121)

    # Convert the RGB values to HSV color space
    hsv_blue = cv2.cvtColor(np.uint8([[rgb_blue]]), cv2.COLOR_RGB2HSV)[0][0]

    # Argument order is Hue, Saturation, Value
    # Define the lower and upper bounds for the blue color
    hsv_lower_blue = np.array([hsv_blue[0] - 5, 10, 0])
    hsv_upper_blue = np.array([hsv_blue[0] + 10, 255, 150])

    # Create a mask for the blue color
    mask_blue = cv2.inRange(img, hsv_lower_blue, hsv_upper_blue)

    # Clean up the mask with small erosion then dilation

    # Create a structuring element for erosion
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # Apply erosion on the mask_blue
    mask_blue_eroded = cv2.erode(mask_blue, kernel_erode, iterations=5)

    # Create a structuring element for dilation
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    # Apply dilation on the eroded mask_blue
    mask_blue_processed = cv2.dilate(mask_blue_eroded, kernel_dilate, iterations=3)

    return mask_blue_processed

## @ingroup Group2
def findPedXPos(mask):
    """
    Returns the centroid of the largest contour in the mask.

    Args:
        mask: The mask to be processed.

    Returns:
        The centroid of the largest contour in the mask.

    Raises:
        None
    """

    # Find the contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Find the largest contour in the mask
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the centroid of the largest contour
    M = cv2.moments(largest_contour)
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])

    return centroid_x

## @ingroup Group3
def skyScanner(img):
    '''
    @brief Returns a debugging overlay for finding the corner point of the sky line, along with the point if found
    If there is no valid skyline then it returns None

    This extrapolates the end of the road based on the instersection of the treeline skybox 
    at the end of the road for each of four sides on the outer track. The points of interest are
    drawn and overlaid on the exported image for debugging, along with the extrapolated point.  
    The point indicates the road terminus from the perspective of the camera near a corner of the track

    @param img (numpy.ndarray): The input image to be processed.

    @returns A tuple containing two elements:
    - overlay_img (numpy.ndarray): An image overlayed with debugging information for finding the corner point of the skyline.
    - extension_pixel (tuple): The coordinates of the extension pixel.
    - None if no valid skyline is found

    @raise None
    '''

    largest_contour_mask = getSkyMask(img)

    if largest_contour_mask is None or type(largest_contour_mask) is tuple:
        return None, None

    # Create a color version of the mask
    color_mask = cv2.cvtColor(largest_contour_mask, cv2.COLOR_GRAY2BGR)

    top_pixel, bottom_pixel = getSkylinePixels(color_mask)

    if (bottom_pixel[0] < 30):
        return color_mask, None
    
    # Calculate the slope of the line connecting the two points
    slope = (bottom_pixel[1] - top_pixel[1]) / (bottom_pixel[0] - top_pixel[0])
    if slope == 0 or math.isnan(slope):
        return color_mask, None
    
    # Draw red dot on top_pixel and green dot on bottom_pixel
    cv2.circle(color_mask, top_pixel, 5, (0, 0, 255), -1)
    cv2.circle(color_mask, bottom_pixel, 5, (0, 255, 0), -1)
    
    # extrapolate to find the terminus of the line
    end_of_path = find_end_of_path(color_mask, top_pixel, bottom_pixel,slope)

    if not isinstance(end_of_path,tuple) :
        print (type(end_of_path))
        return color_mask, None

    # Extend the line to the center of the road from the control viewpoint
    extension_pixel = None
    valid_end_of_path = isinstance(end_of_path, tuple) and isinstance(end_of_path[0], int) and isinstance(end_of_path[1], int)
    if valid_end_of_path:
        extension_pixel = (int(end_of_path[0]-30), int(end_of_path[1]+30))

        # Draw a blue circle at the end of the path
        cv2.circle(color_mask, end_of_path, 5, (255, 0, 0), -1)

    # Create a black canvas of the same size as the original image
    canvas = np.zeros_like(img)
    # Resize the thresh_color image by pasting it onto the black canvas
    canvas[0:color_mask.shape[0], 0:color_mask.shape[1]] = color_mask
    thresh_color_resized = canvas

    # Overlay the two images using addWeighted function
    overlay_img = cv2.addWeighted(thresh_color_resized, 0.6, img, 0.4, 0)

    if valid_end_of_path:
        # Draw a green circle 20 pixels to the left of the blue one with a larger diameter
        cv2.circle(overlay_img, extension_pixel, 20, (0, 255, 0), -1)

    return overlay_img, extension_pixel

def getSkyMask(img):
    '''
    @brief Extract the sky mask from an image.

    Applying color filters for blue and white colors to get sky and cloud. Truncate image to the top 1/3 of the image.
    It first converts the input image to HSV color space and then creates masks for blue and white colors. 
    The masks are combined and returned as the final sky mask.


    @param img (numpy.ndarray): The input image in BGR format, expect a competition image, full resolution.

    @returns The combined sky mask (numpy.ndarray).

    @raise
    '''


    cut = img[0:SKY_SCANNER_CUTOFF, :]
    cut = cv2.cvtColor(cut, cv2.COLOR_BGR2HSV)

    # Define the RGB values for the blue color of the sky
    rgb_blue = (138, 155, 209)

    # Convert the RGB values to HSV color space
    hsv_blue = cv2.cvtColor(np.uint8([[rgb_blue]]), cv2.COLOR_RGB2HSV)[0][0]

    # Argument order is Hue, Saturation, Value
    # Define the lower and upper bounds for the blue color
    hsv_lower_blue = np.array([hsv_blue[0] - 2, 10, 150])
    hsv_upper_blue = np.array([hsv_blue[0] + 2, 160, 255])

    # Create a mask for the blue color
    mask_blue = cv2.inRange(cut, hsv_lower_blue, hsv_upper_blue)

    # Define the lower and upper bounds for the white color
    hsv_lower_white = np.array([0, 0, 200])
    hsv_upper_white = np.array([255, 20, 255])

    # Create a mask for the white color to account for clouds etc
    mask_white = cv2.inRange(cut, hsv_lower_white, hsv_upper_white)

    # Merge the blue and white masks
    mask_combined = cv2.bitwise_or(mask_blue, mask_white)

    # Define the kernel for dilation and erosion
    kernel = np.ones((3, 3), np.uint8)

    # Perform erosion
    processed_mask = cv2.erode(mask_combined, kernel, iterations=2)

    # Perform dilation
    dilated_mask = cv2.dilate(processed_mask, kernel, iterations=3)

    return dilated_mask

def getLargestContour(mask_blue):
    '''
    @brief Find the largest contour in a mask and return a mask with only the largest contour.

    This function takes a mask as input, finds all the contours in it, and returns a new mask containing only
    the largest contour. If no contours are found, it returns None and 0. The output mask is Gaussian blurred
    to reduce noise.

    @param mask_blue (numpy.ndarray): The input mask.

    @returns The mask with the largest contour, Gaussian blurred (numpy.ndarray).

    @raise
    '''

    # Find the contours in the mask
    contours, hierarchy = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there are none, flag that
    if len(contours) == 0:
        return None, 0
    
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the largest contour on the mask
    mask_with_contour = np.zeros_like(mask_blue)
    cv2.drawContours(mask_with_contour, [largest_contour], 0, (255, 255, 255), -1)
    # Apply Gaussian blur to the mask_with_contour to help image processing
    mask_with_contour_blurred = cv2.GaussianBlur(mask_with_contour, (3, 3), 0)  

    return mask_with_contour_blurred 

def getSkylinePixels(color_mask):
    '''
    @brief Find two pixels to represent the right side of the skyline above trees

    Takes a skymask and finds the insersection with contour on the right side at two heights
    
    @param color_mask (numpy.ndarray): The input color mask.

    @returns A tuple containing the coordinates of the top_pixel and bottom_pixel (tuple, tuple).

    @raise
    '''

    # Get the width of the image
    width = color_mask.shape[1]

    # Find the first white pixel on row from the top, moving left from the right side
    top_pixel_row = 50
    top_pixel_col = np.argmax(color_mask[top_pixel_row, ::-1, 0] == 255)
    top_pixel_col = width - top_pixel_col

    # Find the first white pixel on row from the top, moving left from the right side
    bottom_pixel_row = 270
    bottom_pixel_col = np.argmax(color_mask[bottom_pixel_row, ::-1, 0] == 255)
    bottom_pixel_col = width - bottom_pixel_col

    # package the results as pixels
    top_pixel = (top_pixel_col, top_pixel_row)
    bottom_pixel = (bottom_pixel_col, bottom_pixel_row)

    return top_pixel, bottom_pixel

def find_end_of_path(color_mask, top_pixel, bottom_pixel, slope):
    '''
    @brief Find the end of the path by following a line based on two points in a given color mask.

    This function takes a color mask, two points, and a slope as input. It finds the terminus of the line
    Searching for valid contour in white to either side of the line to account for noise.  The end of path is when 
    there are no white pixels within a threshold distance of the line.

    @param color_mask (numpy.ndarray): The input color mask.
    @param top_pixel (tuple): The coordinates of the top pixel.
    @param bottom_pixel (tuple): The coordinates of the bottom pixel.
    @param slope (float): The slope of the line formed by the top and bottom pixels.

    @returns The coordinates of the end of the path (tuple).

    @raise
    '''
    # Calculate the intercept of the line
    intercept = top_pixel[1] - slope * top_pixel[0]

    # Define the width to check for white pixels on either side of the line
    path_width = 10

    # Start at the bottom pixel and move down the image
    current_row = bottom_pixel[1] + 1
    found_white_pixel = True

    #Iterate down row by row finding a new white pixel
    while found_white_pixel:
        found_white_pixel = False
        current_col = (current_row - intercept) / slope

        # Check if the current column is out of bounds - terminate then
        if current_col < 0 or current_col >= color_mask.shape[1] or math.isnan(current_col):
            return (0,0), (0,0)
        else :
            current_col = int(current_col)

        # Check offset column is also in bounds or terminate
        for col_offset in range(-path_width, 1+path_width):           
            col = max(current_col + col_offset, 0) # Make sure col is not negative
            col = min(col, 1279) # Make sure col is not out of bounds
            if color_mask[current_row, col, 0] == 255:
                found_white_pixel = True
                break
        # Once we have no white pixel, the vertex terminus is the row and col
        if not found_white_pixel:
            break
        current_row += 1

    return (current_col, current_row)

def lineVision(img):
    """
    Filter an image for the road surface.

    Args:
        img: The input image in BGR color space.

    Returns:
        mask_white: A binary mask highlighting the road surface of the image
    """

    # Define the RGB values for white
    rgb_white = (255, 255, 255)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = removeSky(hsv)


    # Convert the RGB values to HSV color space
    hsv_gray = cv2.cvtColor(np.uint8([[rgb_white]]), cv2.COLOR_RGB2HSV)[0][0]

    # Argument order is Hue, Saturation, Value
    # Define the lower and upper bounds for the grey color
    hsv_lower_gray = np.array([hsv_gray[0] - 1, 0, 65])
    hsv_upper_gray = np.array([hsv_gray[0] + 1, 40, 95])

    # Create a mask for the grey color
    mask_gray = cv2.inRange(hsv, hsv_lower_gray, hsv_upper_gray)

    return mask_gray

def thresholdTruck(img):
    '''
    @brief Threshold for the black windows and tires of the competition truck, return the mask

    HSV filtering is used to find black areas of the image.  The top right of the image is also blocked out from thresholding
    to ignore the truck when in this area of the screen

    @param img (numpy.ndarray): A raw color image from the competition

    @returns mask (numpy.ndarray): A thresholded mask for the truck
    '''  
    # Define the RGB values for white/gray/black
    rgb_truck = (255, 255, 255)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Convert the RGB values to HSV color space
    hsv_gray = cv2.cvtColor(np.uint8([[rgb_truck]]), cv2.COLOR_RGB2HSV)[0][0]

    # Argument order is Hue, Saturation, Value
    # Define the lower and upper bounds for the grey color
    hsv_lower_gray = np.array([hsv_gray[0] - 1, 0, 0])
    hsv_upper_gray = np.array([hsv_gray[0] + 1, 1, 25])

    # Create a mask for the Truck color
    mask_gray = cv2.inRange(hsv, hsv_lower_gray, hsv_upper_gray)

    # Black out the top right quarter of the image
    h, w = img.shape[:2]
    mask_gray[:int(h/1.8), int(w//1.2):] = 0
    mask_gray[:440, 600:1240] = 0

    return mask_gray 
