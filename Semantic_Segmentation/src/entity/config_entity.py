from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

@dataclass
class DataIngestionConfig:
    kaggle_json_path: Path
    dest_dir: Path
    extract_dir: Path
    source_URL: str
    unzip: bool
    zip_file_name: str

@dataclass
class SegmentationTransformsConfig:
    size: Tuple[int, int]
    mean: List[float, float, float]
    std: List[float, float, float]
    num_classes: int

@dataclass
class SegmentationDatasetConfig:
    root: Path
    img_folder_name: str
    label_folder_name: str

@dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool

@dataclass
class AttentionUNetConfig:
    in_channels: int
    base_channels: int
    upsample_size: Tuple[int, int] = (256, 512)

@dataclass
class SimpleFCNConfig:
    # in_channels: int
    # base_channels: int
    # upsample_size: Tuple[int, int] = (256, 512)
    pass

@dataclass
class DeepLabV3PConfig:
    # in_channels: int
    # base_channels: int
    # upsample_size: Tuple[int, int] = (256, 512)
    pass

@dataclass
class ModelConfig:
    model_name: str
    num_classes: int

@dataclass
class TrainingConfig:
    checkpoint_dir: Path
    num_epochs: int
    batch_size: int
    learning_rate: float
    num_classes: int

@dataclass
class MetricLoggerConfig:
    log_dir: Path
    model_name: str
    model_name: str