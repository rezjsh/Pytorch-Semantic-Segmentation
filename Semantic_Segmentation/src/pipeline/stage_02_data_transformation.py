from src.components.data_transformation import SegmentationTransforms
from src.entity.config_entity import SegmentationTransformsConfig
from src.utils.logging_setup import logger


class DataTransformationPipeline:
    def __init__(self, config: SegmentationTransformsConfig):
        """
        Initialize the segmentation transforms pipeline.

        Args:
            config (SegmentationTransformsConfig): Configuration for segmentation transforms.
        """
        self.config = config
        

    def run_pipeline(self):
        """
        Run the segmentation transforms pipeline.

        Returns:
            SegmentationTransforms: The segmentation transforms object.
        """
        logger.info("Running data transformation pipeline")
        transforms_config = self.config.get_segmentation_transforms_config()
        transforms = SegmentationTransforms(config=transforms_config) 
        logger.info("Data transformation pipeline completed")
        return transforms
