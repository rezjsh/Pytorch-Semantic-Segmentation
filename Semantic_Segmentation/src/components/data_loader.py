from torch.utils.data import DataLoader
from src.entity.config_entity import DataLoaderConfig
from src.components.dataset import SegmentationDataset
from src.components.data_transformation import SegmentationTransforms
from src.utils.logging_setup import logger



class SegmentationDataLoader:
    """
    a class to create segmentation data loaders
    based on string identifiers and configuration parameters.
    """
    def __init__(self, config: DataLoaderConfig, train_dataset: SegmentationDataset = None, val_dataset: SegmentationDataset = None):
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def create_dataloaders(self):
        logger.info("Creating dataloaders")
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers, pin_memory=self.config.pin_memory
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, pin_memory=self.config.pin_memory
        )

        logger.info("DataLoaders created")
        return train_loader, val_loader
