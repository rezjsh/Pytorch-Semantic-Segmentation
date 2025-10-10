from src.models import AttentionUNet, DeepLabV3P, SimpleFCN
from src.utils.logging_setup import logger


class ModelFactory:
    """
    Factory class to create segmentation models
    based on string identifiers and configuration parameters.
    """

    @staticmethod
    def create_model(model_name: str, num_classes: int):
        """
        Instantiate a segmentation model based on the model name.

        Args:
            model_name (str): Name of the model to create. Supported:
                'ATTENTIONUNET', 'DEEPLABV3P', 'SIMPLEFCN'.
            num_classes (int): Number of output segmentation classes.

        Returns:
            nn.Module: Initialized model instance.

        Raises:
            ValueError: If the model_name is unknown.
        """
        model_name_upper = model_name.upper()
        logger.info(f"Creating model '{model_name_upper}' with {num_classes} classes.")

        if model_name_upper == 'ATTENTIONUNET':
            return AttentionUNet(num_classes=num_classes)
        elif model_name_upper == 'DEEPLABV3P':
            return DeepLabV3P(num_classes=num_classes)
        elif model_name_upper == 'SIMPLEFCN':
            return SimpleFCN(num_classes=num_classes)
        else:
            logger.error(f"Unknown model name requested: {model_name_upper}")
            raise ValueError(f"Unknown model name: {model_name}")
