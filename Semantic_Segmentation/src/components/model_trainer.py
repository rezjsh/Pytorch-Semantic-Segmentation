import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Semantic_Segmentation.src.entity.config_entity import TrainingConfig
from src.utils.logging_setup import logger
from src.utils.helpers import calculate_iou
from src.modules.MetricLogger import MetricLogger
from src.utils.helpers import device

class Trainer:
    """
    Trainer class implementing training and validation loops with 
    checkpointing, metric logging and easy integration with config.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        self.config = config
        self.device = device if device else torch.device(config.device)
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.current_epoch = 0

        self.criterion = nn.CrossEntropyLoss(ignore_index=255).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.best_iou = 0.0

        logger.info(f"Trainer initialized on device: {self.device}")
        self.logger = MetricLogger()

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        with tqdm(self.train_loader, desc=f"Train Epoch {self.current_epoch}", leave=False) as loop:
            for data, targets in loop:
                targets = torch.clamp(targets, 0, self.config.num_classes - 1)
                data, targets = data.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(data)

                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Epoch {self.current_epoch} Training Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def validate_epoch(self) -> float:
        self.model.eval()
        total_iou = 0.0

        with tqdm(self.val_loader, desc=f"Validation Epoch {self.current_epoch}", leave=False) as loop:
            for data, targets in loop:
                targets = torch.clamp(targets, 0, self.config.num_classes - 1)
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                preds = torch.argmax(outputs, dim=1)

                iou = calculate_iou(preds, targets, self.config.num_classes)
                total_iou += iou
                loop.set_postfix(mIoU=f"{iou:.4f}")

        avg_iou = total_iou / len(self.val_loader)
        logger.info(f"Epoch {self.current_epoch} Validation mIoU: {avg_iou:.4f}")
        return avg_iou

    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_iou': self.best_iou,
        }

        last_path = os.path.join(self.config.checkpoint_dir, f'{self.config.model_name}_last.pth')
        torch.save(checkpoint, last_path)
        logger.info(f"Saved checkpoint: {last_path}")

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, f'{self.config.model_name}_best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved BEST model checkpoint to: {best_path}")

    def run(self):
        logger.info(f"Starting training for {self.config.num_epochs} epochs with model: {self.config.model_name}")

        for epoch in range(1, self.config.num_epochs + 1):
            self.current_epoch = epoch

            train_loss = self.train_epoch()
            val_iou = self.validate_epoch()

            self.logger.add_entry(train_loss, val_iou)

            is_best = val_iou > self.best_iou
            if is_best:
                self.best_iou = val_iou

            self.save_checkpoint(is_best)

        self.logger.plot_metrics()
        logger.info("Training complete.")
