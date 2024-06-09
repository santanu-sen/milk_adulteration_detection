import numpy as np
import os
import joblib
from PIL import Image
from detection.config.configuration import get_training_config


class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = joblib.load(os.path.join("artifacts","training", "model.joblib"))

        imagename = self.filename
        test_image = Image.open(imagename)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        trainingConfig = get_training_config()


        if result[0] > trainingConfig.params_threshold :
            prediction = 'Adulterated'
            return [{ "image" : prediction}]
        else:
            prediction = 'Not Adulterated'
            return [{ "image" : prediction}]
