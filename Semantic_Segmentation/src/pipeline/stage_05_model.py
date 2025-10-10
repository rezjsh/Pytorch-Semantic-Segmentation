from torch.nn import Module
from Semantic_Segmentation.src.components.model import SegmentationModel
from Semantic_Segmentation.src.entity.config_entity import ModelConfig
from typing import Optional
from src.utils.logging_setup import logger


class ModelPipeline:
    """
    Pipeline class to configure, build, and hold a segmentation model instance
    dynamically using the ModelFactory and SegmentationModelManager.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the segmentation model pipeline.

        Args:
            config (ModelConfig): Configuration object containing model parameters.
        """
        self.config = config.get_model_config()
        self.model: Optional[Module] = None

    def build_model(self) -> Module:
        """
        Build the segmentation model via the ModelFactory using given config.

        Returns:
            torch.nn.Module: Instantiated segmentation model.
        """
        logger.info(
            f"ModelPipeline building model '{self.config.model_name}' "
            f"with {self.config.num_classes} classes and config: {self.config}"
        )
        self.model = SegmentationModel(self.config).build()
        logger.info(f"Model '{self.config.model_name}' created successfully.")
        return self.model
