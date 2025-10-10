from typing import Optional, Callable
from src.utils.logging_setup import logger
from src.entity.config_entity import SegmentationDatasetConfig
from src.components.dataset import SegmentationDataset


class DatasetPipeline:
    def __init__(
        self,
        config: SegmentationDatasetConfig,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the Cityscapes dataset pipeline.

        Args:
            config (SegmentationDatasetConfig): Configuration for the dataset.
            transform (Callable, optional): Transform function for images and labels. Defaults to None.
        """
        self.config = config
        self.transform = transform

    def run_pipeline(self):
        """
        Run the dataset pipeline.
        Returns:
            SegmentationDataset: The dataset object.
        """
        logger.info("Running dataset pipeline")
        get_dataset_preparation_config = self.config.get_dataset_preparation_config()
        train_dataset = SegmentationDataset(config=get_dataset_preparation_config, transform=self.transform, split='train')
        val_dataset = SegmentationDataset(config=get_dataset_preparation_config, transform=self.transform, split='val')
        logger.info("Dataset pipeline completed")
        return train_dataset, val_dataset
