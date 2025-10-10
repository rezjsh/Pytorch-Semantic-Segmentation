from typing import Tuple
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from src.entity.config_entity import SegmentationTransformsConfig
from src.utils.logging_setup import logger


class SegmentationTransforms:
    def __init__(self, config: SegmentationTransformsConfig , size: Tuple[int, int]):
        """
        Compose image and target transforms for segmentation.
        Resizes images and targets to the given size.
        Normalizes images with ImageNet mean/std.
        """
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize(self.config.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std),
        ])

    def __call__(self, image: Image.Image, target: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.transform(image)
        target = transforms.functional.resize(target, self.config.size, interpolation=transforms.InterpolationMode.NEAREST)

        target_np = np.array(target, dtype=np.int64).squeeze()
        target = torch.from_numpy(target_np)

        # Filter invalid target values (valid IDs 0-20, ignore 255)
        invalid_mask = (target > self.config.num_classes) & (target != 255)
        target[invalid_mask] = 255

        if target.dim() != 2:
            if target.dim() == 3:
                logger.warning("Target mask is 3D; using first channel assuming identical RGB labels.")
                target = target[..., 0]
            if target.dim() != 2:
                raise ValueError(f"Target mask dimension is {target.dim()} after processing; expected 2.")
        return image, target
