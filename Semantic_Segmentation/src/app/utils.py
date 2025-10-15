# src/app/utils.py

import numpy as np
import io
from PIL import Image
from base64 import b64encode


def get_simple_colormap(num_classes: int) -> np.ndarray:
    """Generates a simple, distinct colormap for up to N classes."""
    # This is a placeholder. For production, use the standard Cityscapes colormap.
    cmap = np.zeros((256, 3), dtype=np.uint8)
    
    # Generate some distinct colors
    for i in range(num_classes):
        r = (i * 30 + 50) % 256
        g = (i * 50 + 100) % 256
        b = (i * 70 + 150) % 256
        cmap[i] = [r, g, b]
        
    # Set the ignore index (255) to black
    cmap[255] = [0, 0, 0]
    return cmap

# Generate the colormap based on your project's num_classes (21 in params.yaml)
CLASS_COLORMAP = get_simple_colormap(num_classes=21)

def colorize_mask(mask: np.ndarray, num_classes: int) -> Image.Image:
    """
    Converts a class index mask (H, W) to a colorized PIL Image (H, W, 3).
    """
    # The mask contains class indices (0 to 20 or 255)
    # Map index values to colors using the global colormap
    color_mask = CLASS_COLORMAP[mask]
    return Image.fromarray(color_mask).convert('RGB')

def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 encoded string for embedding in HTML."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"