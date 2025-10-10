
from Semantic_Segmentation.src.components.model_trainer import Trainer
from src.entity.config_entity import TrainingConfig
from src.utils.logging_setup import logger

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config.get_model_trainer_config()

    def run_pipeline(self, model, train_loader, val_loader):
        logger.info("Running Model Trainer pipeline")
        trainer = Trainer(config=self.config, model=model, train_loader=train_loader, val_loader=val_loader)
        trainer.run()
        logger.info("Model Trainer pipeline completed")