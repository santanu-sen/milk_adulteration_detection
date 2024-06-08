import os
import xgboost as xgb
import joblib
from pathlib import Path
from detection.entity.config_entity import PrepareBaseModelConfig



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config


    
    
    
    def get_base_model(self):

        self.model = xgb.XGBClassifier()

        self.save_model(self, path=self.config.base_model_path, model=self.model)


    
    @staticmethod
    def _prepare_full_model(model):
        
        full_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
        )

        self.save_model(self, path=self.config.updated_base_model_path, model=self.full_model)

    
    @staticmethod
    def save_model(self, path: Path, model: xgb):
        joblib.dump( model, path)



