# src/app/prediction_service.py

import torch
import os
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from src.config.configuration import ConfigurationManager
from src.components.model import SegmentationModel
from src.components.data_transformation import SegmentationTransforms
from src.utils.helpers import device 
from src.utils.logging_setup import logger
from src.app.utils import colorize_mask
class PredictionService:
    """
    Singleton class to load and manage the segmentation model for inference.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info("Initializing PredictionService: Loading configurations, model, and transforms.")
        self.model = None
        
        try:
            config_manager = ConfigurationManager()
            
            # --- 1. Load Model Architecture ---
            model_config = config_manager.get_model_config()
            model_manager = SegmentationModel(config=model_config)
            self.model = model_manager.build()
            self.device = device if device else torch.device("cpu") # Use a defined device or fallback to CPU
            self.model.to(self.device)
            
            # --- 2. Load Trained Weights ---
            trainer_config = config_manager.model_trainer_config()
            # ASSUMPTION: The best model checkpoint is saved here
            best_model_path = os.path.join(trainer_config.checkpoint_dir, f'{model_config.model_name}_best.pth')

            if not os.path.exists(best_model_path):
                raise FileNotFoundError(f"Best checkpoint not found: {best_model_path}")

            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval() # Set model to evaluation mode
            
            logger.info(f"Model {model_config.model_name} loaded successfully from {best_model_path}")

            # --- 3. Load Data Transforms for Preprocessing ---
            transforms_config = config_manager.get_segmentation_transforms_config()
            # Reuse your existing transforms class for input image preprocessing
            self.transform = SegmentationTransforms(config=transforms_config, size=transforms_config.size)
            self.num_classes = transforms_config.num_classes

            self._initialized = True
            logger.info("PredictionService initialization complete and model is ready.")
        
        except Exception as e:
            logger.error(f"Failed to initialize PredictionService. Is the checkpoint available? Error: {e}")
            self.model = None
            self._initialized = True # Mark as initialized to prevent re-attempts

    @torch.no_grad()
    def predict(self, image_file_path: str) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Performs segmentation inference on a single image file."""
        if self.model is None:
             logger.error("Model is not loaded. Cannot perform prediction.")
             return None, None

        try:
            # 1. Load Original Image
            original_img = Image.open(image_file_path).convert('RGB')
            original_size = original_img.size

            # 2. Preprocess
            # Create a dummy target to reuse the SegmentationTransforms logic (it expects two inputs)
            dummy_target = Image.fromarray(np.zeros(original_size[::-1], dtype=np.uint8)).convert('L')
            input_tensor, _ = self.transform(original_img, dummy_target)
            
            # Add batch dimension and move to device
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            # 3. Inference
            output = self.model(input_batch)

            # 4. Post-process (get class indices)
            # Find the class with the highest probability: (B, C, H, W) -> (B, H, W)
            _, predicted_mask_tensor = torch.max(output, 1)
            predicted_mask = predicted_mask_tensor.squeeze(0).cpu().numpy() # (H, W) numpy array

            # 5. Colorize and Resize Mask
            # Convert class indices to a colorized image mask
            mask_img = colorize_mask(predicted_mask, self.num_classes)
            
            # Resize mask back to the original image dimensions for display
            mask_img = mask_img.resize(original_size, resample=Image.NEAREST)

            logger.info("Prediction successful.")
            return original_img, mask_img 
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None, None