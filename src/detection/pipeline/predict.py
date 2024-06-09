import numpy as np
import joblib
from PIL import Image
import yaml
from pathlib import Path
import cv2
from detection.components.data_ingestion import DataIngestion



def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


class PredictionPipeline:
    def __init__(self, filename, directory):
        self.filename = filename
        self.directory = directory
        self.config_path = Path('C:\\Users\\admin\\Desktop\\JI\\Project\\capstone\\milk_adulteration_detection\\params.yaml')
    
    
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

    def preprocess_image(self):
        image = DataIngestion.read_image(self, self.directory)
        img_h, img_g = DataIngestion.segment_strip1(self, image)
        
        img_cv = cv2.cvtColor(img_g, cv2.COLOR_RGB2BGR)
        
        # Extract features from RGB
        r, g, b = cv2.split(img_cv)
        average_r = r.mean()
        average_g = g.mean()
        average_b = b.mean()

        # Convert RGB image to HSV
        hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        average_h = h.mean()
        average_s = s.mean()
        average_v = v.mean()

        # Convert RGB image to LAB
        lab_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b_lab = cv2.split(lab_img)
        average_l = l.mean()
        average_a = a.mean()
        average_b_lab = b_lab.mean()

        # Combine all features into a single array
        features = np.array([average_r, average_g, average_b, average_h, average_s, average_v, average_l, average_a, average_b_lab])

        return features


    def predict(self):
        # Load the model
        model = joblib.load(Path('C:\\Users\\admin\\Desktop\\JI\\Project\\capstone\\milk_adulteration_detection\\artifacts\\training\\model.joblib'))

        # Read the threshold from the config file
        config = read_params(self.config_path)
        threshold = config['THRESHOLD']

        # Preprocess the image
        image_features = self.preprocess_image()

        # Predict using the model
        probabilities = model.predict_proba(image_features.reshape(1, -1))  # Assuming the model expects a 2D array

        # Check if the probability of the positive class (e.g., index 1) exceeds the threshold
        result = probabilities[0][1] > threshold
        
        # Interpret the result
        if result:
            prediction = 'Adulterated'
        else:
            prediction = 'Not Adulterated'

        return [{"image": prediction}]

