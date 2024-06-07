import os
import urllib.request as request
from detection import logger
from detection.utils.common import get_size
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from detection.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def read_image(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    
    def pad_to_size(self, image, desired_height, desired_width):
        # Get the current size of the image
        height, width = image.shape[:2]

        # Calculate the amount of padding needed
        pad_height = max(0, desired_height - height)
        pad_width = max(0, desired_width - width)

        # Calculate the padding amounts for top, bottom, left, and right sides
        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad

        # Pad the image with zeros
        padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return padded_image


    def find_contours1(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a threshold or any other preprocessing if needed
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours based on their area in descending order and select top two contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2] if len(contours) >= 2 else contours

        # Filter contours with an area less than 18000
        contours = [contour for contour in contours if cv2.contourArea(contour) >= 10000]

        # Initialize variables for left and right contours
        left_cropped_image = np.zeros((200, 200, 3), dtype=np.uint8)
        right_cropped_image = np.zeros((200, 200, 3), dtype=np.uint8)

        if len(contours) == 1:
            left_contour = contours[0]
            x1, y1, w1, h1 = cv2.boundingRect(left_contour)
            # Crop left region
            left_cropped_image = image[y1:y1+h1, x1:x1+w1]
        elif len(contours) == 2:
            # Sort contours by x-coordinate
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
            left_contour = contours[0]
            right_contour = contours[1]
            x1, y1, w1, h1 = cv2.boundingRect(left_contour)
            x2, y2, w2, h2 = cv2.boundingRect(right_contour)
            # Crop left and right regions
            left_cropped_image = image[y1:y1+h1, x1:x1+w1]
            right_cropped_image = image[y2:y2+h2, x2:x2+w2]

            # Pad the cropped images to 200x200
            left_cropped_image = self.pad_to_size(left_cropped_image, 200, 200)
            right_cropped_image = self.pad_to_size(right_cropped_image, 200, 200)

            #cropping to 50x50 to get only the center portion of the patch
            left_cropped_image = left_cropped_image[75:125,75:125,:]
            right_cropped_image = right_cropped_image[75:125,75:125,:]

            if left_cropped_image.all() == 0:
                left_cropped_image = None

            if right_cropped_image.all() == 0:
                right_cropped_image = None

        return left_cropped_image, right_cropped_image

    def segment_strip1(self, image):

        #Crop image to keep only black portion of machine in image
        image_modified = image[:2000, 1000:2600]

        #Make a copy of image
        image_copy = image_modified.copy()


        # Convert the difference image to HSV
        img_hsv = cv2.cvtColor(image_modified, cv2.COLOR_BGR2HSV)

        # Split the HSV image into individual channels
        hue, _, _ = cv2.split(img_hsv)

        #Apply thresholding
        hue[(hue<60) | (hue>120)] = 0

        # Create a square structuring element for morphological operations
        structuring_element = np.ones((11, 11), np.uint8)


        # Perform erosion followed by dilation to remove noise
        eroded_image = cv2.erode(hue, structuring_element, iterations=5)
        dilated_image = cv2.dilate(eroded_image, structuring_element, iterations=3)

        # Apply the mask to the original image
        result_image = cv2.bitwise_and(image_copy, image_copy, mask=dilated_image)

        img_h, img_g = self.find_contours1(result_image)

        return img_h, img_g

    
    def read_save(self):
        # Traverse through all subdirectories and read images
        for dirpath, dirnames, filenames in os.walk(rf'{self.config.root_folder}'):
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    # Construct the full path to the image
                    image_path = os.path.join(dirpath, filename)

                    # Read the image (using OpenCV in this example)
                    image = self.read_image(image_path)

                    img_h2o2, img_glucose = self.segment_strip1(image)

                    if img_h2o2 is not None:

                        h_dir = self.config.local_data_file_h202 + image_path.split('\\')[-2] + '\\'

                        if not os.path.exists(h_dir):
                            os.makedirs(h_dir)

                        # save the processed image to another location
                        cv2.imwrite(h_dir + image_path.split('\\')[-3] + '_' + filename[:-4] + '.png', cv2.cvtColor(img_h2o2, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])

                    if img_glucose is not None:
                        g_dir = self.config.local_data_file_gluscose + image_path.split('\\')[-2] + '\\'

                        if not os.path.exists(g_dir):
                            os.makedirs(g_dir)

                        # save the processed image to another location
                        cv2.imwrite(g_dir + image_path.split('\\')[-3] + '_' + filename[:-4] + '.png', cv2.cvtColor(img_glucose, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])


    