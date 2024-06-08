import os
import urllib.request as request
import xgboost as xgb
import joblib
from pathlib import Path
import time
import pandas as pd
import cv2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from detection.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = joblib.load(
            self.config.updated_base_model_path
        )
    
    #function to load all images as data

    def load_data(self, file_path, threshold  ):

        file_path = self.config.training_data
        threshold = self.config.params_threshold

        folders = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1', '1.5', '2', '2.5','3', '4', '5', '6', '7', '8']

        #Define dataframe in which the image is saved

        # Define column names
        columns = ['Red Channel', 'Green Channel', 'Blue Channel', 'Hue', 'Saturation', 'Value', 'Lightness', 'channel a',  'channel b', 'Target']

        # Create an empty DataFrame with columns
        df = pd.DataFrame(columns=columns)

        # Iterate through folders
        for folder in folders:
            folder_path = str(file_path) + folder

            # Iterate through files in the current folder
            for filename in os.listdir(folder_path):
                # Check if the file has an image extension
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    # Construct the full path to the image
                    image_path = os.path.join(folder_path, filename)

                    # Read the image (using OpenCV)
                    img = cv2.imread(image_path)

                    # Split the image into channels
                    r, g, b = cv2.split(img)
                    # Calculate the average of each channel
                    average_r = r.mean()
                    average_g = g.mean()
                    average_b = b.mean()


                    # Convert RGB image to HSV
                    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

                    # Split the HSV image into channels
                    h, s, v = cv2.split(hsv_img)
                    # Calculate the average of each channel
                    average_h = h.mean()
                    average_s = s.mean()
                    average_v = v.mean()

                    # Convert RGB image to LAB
                    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

                    # Split LAB image into components
                    L, a, b = cv2.split(img_lab)
                    # Calculate the average of each channel
                    average_L = L.mean()
                    average_a = a.mean()
                    average_b = b.mean()

                    if float(folder) < threshold:
                        # Append rows one by one
                        new_row = {'Red Channel': average_r, 'Green Channel': average_g, 'Blue Channel': average_b, 'Hue': average_h, 'Saturation': average_s, 'Value': average_v, 'Lightness': average_L, 'channel a': average_a,  'channel b': average_b, 'Target': 0}
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    else:
                        new_row = {'Red Channel': average_r, 'Green Channel': average_g, 'Blue Channel': average_b, 'Hue': average_h, 'Saturation': average_s, 'Value': average_v, 'Lightness': average_L, 'channel a': average_a,  'channel b': average_b, 'Target': 1}
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df['Target'] = df['Target'].astype(int)
        return df

    
    @staticmethod
    def save_model(self, path: Path, model: xgb):
        joblib.dump( model, path)

    def train(self, df ):

        df_shuffled = df.sample(frac=1, random_state=42)

        # 'Target' column is the target variable
        y = df_shuffled['Target']  # Target variable
        X = df_shuffled.drop(columns=['Target'])  # Features

        # Split the shuffled data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        # Predict on the test data
        y_pred = self.model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print('Accuracy is ', accuracy)




        self.save_model( self,
            path=self.config.trained_model_path,
            model=self.model
        )

