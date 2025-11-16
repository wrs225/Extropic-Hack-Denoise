"""Image loading and preprocessing utilities for 3-bit grayscale images."""

import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional
from PIL import Image


def load_image(
    path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = (256, 256),
    normalize: bool = True,
) -> np.ndarray:
    """
    Load an image and convert to 3-bit grayscale format.
    
    Args:
        path: Path to image file
        target_size: Target size (height, width). If None, keeps original size.
        normalize: If True, normalize to [0, 1]. If False, keep as [0, 255].
    
    Returns:
        Image array as float32, shape (H, W) or (H, W, C) if RGB
    """
    img = Image.open(path)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Resize if needed
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1] if requested
    if normalize:
        img_array = img_array / 255.0
    
    return img_array


def load_images_from_dir(
    directory: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = (256, 256),
    normalize: bool = True,
) -> List[np.ndarray]:
    """
    Load all images from a directory.
    
    Args:
        directory: Path to directory containing images
        target_size: Target size (height, width)
        normalize: If True, normalize to [0, 1]
    
    Returns:
        List of image arrays
    """
    directory = Path(directory)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    images = []
    for file_path in directory.iterdir():
        if file_path.suffix.lower() in image_extensions:
            try:
                img = load_image(file_path, target_size, normalize)
                images.append(img)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
    
    return images


def quantize_to_3bit(image: np.ndarray) -> np.ndarray:
    """
    Quantize image to 3-bit (8 levels: 0-7).
    
    Args:
        image: Image array, assumed to be in [0, 1] range
    
    Returns:
        Quantized image with values in [0, 7]
    """
    # Clamp to [0, 1]
    image = np.clip(image, 0.0, 1.0)
    
    # Quantize to 8 levels (3-bit)
    quantized = (image * 7.0).astype(np.uint8)
    
    return quantized


def dequantize_from_3bit(quantized: np.ndarray) -> np.ndarray:
    """
    Dequantize 3-bit image back to [0, 1] range.
    
    Args:
        quantized: Quantized image with values in [0, 7]
    
    Returns:
        Dequantized image in [0, 1] range as float32
    """
    # Convert to float and normalize
    dequantized = quantized.astype(np.float32) / 7.0
    
    return dequantized


def save_image(image: np.ndarray, path: Union[str, Path], denormalize: bool = True) -> None:
    """
    Save an image array to file.
    
    Args:
        image: Image array, shape (H, W)
        path: Output file path
        denormalize: If True, assume image is in [0, 1] and convert to [0, 255]
    """
    # Ensure 2D
    if len(image.shape) > 2:
        image = image.squeeze()
    
    # Denormalize if needed
    if denormalize:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Save
    img = Image.fromarray(image, mode='L')
    img.save(path)

