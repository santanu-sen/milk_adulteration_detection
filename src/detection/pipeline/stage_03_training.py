from detection.config.configuration import ConfigurationManager
from detection.components.training import Training
from detection import logger



STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        trainingConfig = config.get_training_config()

        
        file_path = trainingConfig.training_data
        threshold = trainingConfig.params_threshold

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        df = training.load_data(file_path , threshold)
        training.train(df)
        
        



if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        
