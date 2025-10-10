from torch.nn import Module
from src.entity.config_entity import ModelConfig
from src.factory.model_factory import ModelFactory
from src.utils.logging_setup import logger


class SegmentationModel:
    """
    Wrapper class to create and hold a segmentation model instance
    selected dynamically using the factory.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the model manager with model selection and number of classes.

        Args:
          config (ModelConfig): Configuration object containing model parameters.
        """
        self.config = config
        self.model: Module = None

    def build(self) -> Module:
        """
        Creates the model instance using the factory.

        Returns:
            torch.nn.Module: The instantiated segmentation model.
        """
        logger.info(f"Building segmentation model: {self.config.model_name} with {self.config.num_classes} classes.")
        self.model = ModelFactory.create_model(self.config.model_name, self.config.num_classes)
        logger.info(f"Model '{self.config.model_name}' created successfully.")
        return self.model
