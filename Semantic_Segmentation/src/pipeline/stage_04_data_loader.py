from src.utils.logging_setup import logger
from src.entity.config_entity import SegmentationDataLoader



class DataLoaderPipeline:
    def __init__(self, config, train_dataset, val_dataset):
        self.config = config.get_data_loader_config()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
    def run_pipeline(self):
        logger.info("Running DataLoader pipeline")
        data_loader = SegmentationDataLoader(
            config=self.config,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset
        )
        train_loader, val_loader = data_loader.create_dataloaders()
        logger.info("DataLoader pipeline completed")
        return train_loader, val_loader
