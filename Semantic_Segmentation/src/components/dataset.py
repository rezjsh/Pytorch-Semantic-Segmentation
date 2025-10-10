from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import torch
import os
import glob
from typing import Optional, Callable, Tuple
from src.entity.config_entity import SegmentationDatasetConfig
from src.utils.logging_setup import logger

class SegmentationDataset(Dataset):
    """
        Custom dataset for semantic segmentation.
    """

    def __init__(
        self,
        config: SegmentationDatasetConfig,
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None,
        split: str = 'train'
    ):
        self.config = config
        self.split = split
        self.transform = transform
        self.img_folder_name = config.img_folder_name
        self.label_folder_name = config.label_folder_name

        self.img_dir = os.path.join(self.config.root, self.split, self.img_folder_name)
        self.label_dir = os.path.join(self.config.root, self.split, self.label_folder_name)

        if not os.path.isdir(self.img_dir) or not os.path.isdir(self.label_dir):
            msg = f"Image or label directory missing in {self.img_dir} or {self.label_dir}."
            logger.error(msg)
            raise RuntimeError(msg)

        self.images = sorted(glob.glob(os.path.join(self.img_dir, '*.png')))
        self.targets = sorted(glob.glob(os.path.join(self.label_dir, '*.png')))

        if not self.images or not self.targets:
            msg = f"No .png images or labels found in {self.img_dir} or {self.label_dir}."
            logger.error(msg)
            raise RuntimeError(msg)

        if len(self.images) != len(self.targets):
            msg = (
                f"Image and label count mismatch in split '{self.split}': "
                f"{len(self.images)} images vs {len(self.targets)} labels."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info(f"Loaded {len(self.images)} samples for split '{self.split}'.")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            image = Image.open(self.images[index]).convert('RGB')
        except (IOError, UnidentifiedImageError) as e:
            logger.error(f"Error loading image file {self.images[index]}: {e}")
            raise e

        try:
            target = Image.open(self.targets[index]).convert('L')
        except (IOError, UnidentifiedImageError) as e:
            logger.error(f"Error loading label file {self.targets[index]}: {e}")
            raise e

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)
