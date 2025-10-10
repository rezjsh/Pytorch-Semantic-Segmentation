import os
import matplotlib.pyplot as plt
from Semantic_Segmentation.src.config.configuration import ConfigurationManager
from Semantic_Segmentation.src.entity.config_entity import MetricLoggerConfig
from src.utils.logging_setup import logger


class MetricLogger:
    """
    Handles accumulating metrics and plotting training loss and validation mIoU.
    """

    def __init__(self):
        self.config: MetricLoggerConfig = ConfigurationManager()
        self.log_dir = self.config.log_dir
        self.model_name = self.config.model_name
        self.history = {'train_loss': [], 'val_miou': []}

    def add_entry(self, train_loss: float, val_miou: float):
        self.history['train_loss'].append(train_loss)
        self.history['val_miou'].append(val_miou)

    def plot_metrics(self):
        epochs = range(1, len(self.history['train_loss']) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['train_loss'], 'r-o', label='Training Loss')
        plt.title(f'{self.model_name} Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        loss_path = os.path.join(self.log_dir, f'{self.model_name}_loss_plot.png')
        plt.savefig(loss_path)
        plt.close()
        logger.info(f"Loss plot saved to {loss_path}")

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['val_miou'], 'b-o', label='Validation mIoU')
        plt.title(f'{self.model_name} Validation mIoU Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.legend()
        miou_path = os.path.join(self.log_dir, f'{self.model_name}_miou_plot.png')
        plt.savefig(miou_path)
        plt.close()
        logger.info(f"Validation mIoU plot saved to {miou_path}")
