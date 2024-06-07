from detection.constants import *
from detection.utils.common import read_yaml, create_directories
from detection.entity.config_entity import DataIngestionConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            root_folder=config.root_folder,
            local_data_file_gluscose=config.local_data_file_gluscose,
            local_data_file_h202=config.local_data_file_h202
        )

        return data_ingestion_config
      