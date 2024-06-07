from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    root_folder: str
    local_data_file_gluscose: str
    local_data_file_h202: str

