import os
import urllib.request as request
import xgboost as xgb
import time
from tensorboardX import SummaryWriter
import shutil
import joblib
from detection.entity.config_entity import PrepareCallbacksConfig


class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(self.config.tensorboard_root_log_dir, f"tb_logs_at_{timestamp}")
        
        # Remove the directory if it already exists
        if os.path.exists(tb_running_log_dir):
            shutil.rmtree(tb_running_log_dir)

        # Create a SummaryWriter for TensorBoard logging
        self.writer = SummaryWriter(log_dir=tb_running_log_dir)

        def tb_callback(env):
            for i in range(len(env.models)):
                self.writer.add_scalar(f'error_{i}', env.evaluation_result_list[i][1], env.iteration)
                self.writer.add_scalar(f'logloss_{i}', env.evaluation_result_list[i][2], env.iteration)

        return tb_callback

    @property
    def _create_ckpt_callbacks(self):
        checkpoint_path = self.config.checkpoint_model_filepath
        
        def ckpt_callback(env):
            joblib.dump(env.model, checkpoint_path)

        return ckpt_callback

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]