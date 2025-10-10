from pathlib import Path
from src.entity.config_entity import DataIngestionConfig, DataLoaderConfig, DeepLabV3PConfig, MetricLoggerConfig, ModelConfig, SegmentationTransformsConfig, SegmentationDatasetConfig, AttentionUNetConfig, SimpleFCNConfig, TrainingConfig
from src.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.utils.helpers import create_directory, read_yaml_file
from src.utils.logging_setup import logger
from src.core.singlton import SingletonMeta


class ConfigurationManager(metaclass=SingletonMeta):
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH, params_file_path: str = PARAMS_FILE_PATH):
        self.config = read_yaml_file(config_file_path)
        self.params = read_yaml_file(params_file_path)


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        logger.info("Getting data ingestion config")
        config = self.config.data_ingestion
        dirs_to_create = [config.dest_dir, config.extract_dir]
        create_directory(dirs_to_create)
        logger.info(f"Directories created: {dirs_to_create}")
        logger.info(f"Data ingestion config: {config}")
        data_ingestion_config = DataIngestionConfig(
            kaggle_json_path=Path(config.kaggle_json_path),
            dest_dir=Path(config.dest_dir),
            extract_dir=Path(config.extract_dir),
            source_URL=config.source_URL,
            unzip=config.unzip,
            zip_file_name=config.zip_file_name)

        logger.info(f"Data ingestion config created: {data_ingestion_config}")
        return data_ingestion_config

    def get_segmentation_transforms_config(self) -> SegmentationTransformsConfig:
        logger.info("Getting segmentation transforms config")
        config = self.params.transforms
        logger.info(f"Segmentation transforms config: {config}")
        segmentation_transforms_config = SegmentationTransformsConfig(
            size=config.size,
            mean=config.mean,
            std=config.std)
        logger.info(f"Segmentation transforms config created: {segmentation_transforms_config}")
        return segmentation_transforms_config

    def get_dataset_preparation_config(self):
        logger.info("Getting dataset preparation config")
        config = self.config.data_preparation
        dirs_to_create = [config.root]
        create_directory(dirs_to_create)
        logger.info(f"Directories created: {dirs_to_create}")
        logger.info(f"Dataset preparation config: {config}")
        dataset_preparation_config = SegmentationDatasetConfig(
            root=Path(config.root),
            img_folder_name=config.img_folder_name,
            label_folder_name=config.label_folder_name
            )
        logger.info(f"Dataset preparation config created: {dataset_preparation_config}")
        return dataset_preparation_config

    def get_data_loader_config(self):
        logger.info("Getting data loader config")
        config = self.config.data_loader
        logger.info(f"Data loader config: {config}")
        data_loader_config = DataLoaderConfig(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        logger.info(f"Data loader config created: {data_loader_config}")
        return data_loader_config

    def get_attention_unet_config(self):
        logger.info("Getting AttentionUNet config")
        params = self.params.attentionUNet_model
        logger.info(f"AttentionUNet config: {params}")
        attention_unet_config = AttentionUNetConfig(
            in_channels=params.in_channels,
            base_channels=params.base_channels,
            upsample_size=params.upsample_size
        )
        logger.info(f"AttentionUNet config created: {attention_unet_config}")
        return attention_unet_config

    def get_simple_fcn_config(self):
        pass

    def get_deeplabv3p_config(self):
        pass

    def get_model_config(self):
        logger.info("Getting model config")
        config = self.params.model
        logger.info(f"Model config: {config}")
        model_config = ModelConfig(
            model_name=config.model_name,
            num_classes=config.num_classes,
        )
        logger.info(f"Model config created: {model_config}")
        return model_config
    

    def model_trainer_config(self):
        logger.info("Getting model trainer config")
        params = self.params.model_trainer
        config = self.config.model_trainer
        logger.info(f"Model trainer config: {params}")
        dirs_to_create = [config.checkpoint_dir]
        create_directory(dirs_to_create)
        logger.info(f"Directories created: {dirs_to_create}")
        model_trainer_config = TrainingConfig(
            checkpoint_dir=Path(config.checkpoint_dir),
            model_name=params.model_name,
            num_epochs=params.num_epochs,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            num_classes=params.num_classes
        )
        logger.info(f"Model trainer config created: {model_trainer_config}")
        return model_trainer_config

    def get_metric_logger_config(self):
        logger.info("Getting metric logger config")
        params = self.params.metric_logger
        config = self.config.metric_logger
        dirs_to_create = [config.log_dir]
        create_directory(dirs_to_create)
        logger.info(f"Directories created: {dirs_to_create}")
        metric_logger_config = MetricLoggerConfig(
            log_dir=Path(config.log_dir),
            model_name=params.model_name
        )
        logger.info(f"Metric logger config created: {metric_logger_config}")
        return metric_logger_config