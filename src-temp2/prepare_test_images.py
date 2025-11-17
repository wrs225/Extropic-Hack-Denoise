"""Generate standardized test images for fair comparison of denoisers."""

import numpy as np
from pathlib import Path
from PIL import Image


def create_simple_image(size: int = 256) -> np.ndarray:
    """Create simple geometric shapes test image."""
    img = np.zeros((size, size), dtype=np.uint8)
    
    center = size // 2
    y, x = np.ogrid[:size, :size]
    
    # Create patterns with 4-bit values (0-15)
    circles = [
        (center, center, size//3, 14),
        (size//4, size//4, size//8, 10),
        (3*size//4, 3*size//4, size//8, 6),
    ]
    
    for cy, cx, r, intensity in circles:
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        img[mask] = intensity
    
    # Rectangles
    img[20:60, 20:80] = 12
    img[size-60:size-20, size-80:size-20] = 8
    
    return img


def create_edges_image(size: int = 256) -> np.ndarray:
    """Create image with sharp edges for testing edge preservation."""
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Vertical edge
    img[:, size//2 - 5:size//2 + 5] = 15
    
    # Horizontal edge
    img[size//2 - 5:size//2 + 5, :] = 15
    
    # Diagonal edge
    for i in range(size):
        for j in range(size):
            if abs(i - j) < 3:
                img[i, j] = 12
    
    # Checkerboard pattern
    checker_size = 32
    for i in range(0, size, checker_size):
        for j in range(0, size, checker_size):
            if (i // checker_size + j // checker_size) % 2 == 0:
                img[i:i+checker_size, j:j+checker_size] = 10
    
    return img


def create_textures_image(size: int = 256) -> np.ndarray:
    """Create image with fine textures for testing texture preservation."""
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Fine checkerboard
    checker_size = 8
    for i in range(0, size, checker_size):
        for j in range(0, size, checker_size):
            if (i // checker_size + j // checker_size) % 2 == 0:
                img[i:i+checker_size, j:j+checker_size] = 15
            else:
                img[i:i+checker_size, j:j+checker_size] = 0
    
    # Add some variation
    img[64:128, 64:128] = 8
    img[128:192, 128:192] = 12
    
    return img


def create_gradients_image(size: int = 256) -> np.ndarray:
    """Create image with smooth gradients for testing gradient smoothness."""
    img = np.zeros((size, size), dtype=np.uint8)
    
    y, x = np.ogrid[:size, :size]
    center = size // 2
    
    # Radial gradient
    r = np.sqrt((x - center)**2 + (y - center)**2)
    r_max = size / 2
    radial = (1.0 - r / r_max) * 15
    radial = np.clip(radial, 0, 15).astype(np.uint8)
    img = np.maximum(img, radial)
    
    # Horizontal gradient
    horizontal = (x / size * 15).astype(np.uint8)
    img[0:size//4, :] = horizontal[0:size//4, :]
    
    # Vertical gradient
    vertical = (y / size * 15).astype(np.uint8)
    img[:, 0:size//4] = np.maximum(img[:, 0:size//4], vertical[:, 0:size//4])
    
    return img


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int = 42) -> np.ndarray:
    """
    Add Gaussian noise to 4-bit image.
    
    Args:
        image: 4-bit image (values 0-15)
        sigma: Noise standard deviation (as fraction of max value)
        seed: Random seed for reproducibility
    
    Returns:
        Noisy image (4-bit, values 0-15)
    """
    np.random.seed(seed)
    
    # Convert to float for noise addition
    img_float = image.astype(np.float32) / 15.0
    
    # Add noise
    noise = np.random.normal(0, sigma, img_float.shape)
    noisy = np.clip(img_float + noise, 0, 1)
    
    # Convert back to 4-bit
    return (noisy * 15).astype(np.uint8)


def save_image_4bit(image: np.ndarray, path: Path):
    """Save 4-bit image to file."""
    # Convert to 8-bit for saving (scale 0-15 to 0-255)
    img_8bit = (image.astype(np.float32) / 15.0 * 255.0).astype(np.uint8)
    Image.fromarray(img_8bit, mode='L').save(path)


def main():
    """Generate test images."""
    print("="*70)
    print("Generating Standardized Test Images")
    print("="*70)
    
    # Setup directories
    base_dir = Path("data/common_test")
    clean_dir = base_dir / "clean"
    noisy_dir = base_dir / "noisy"
    
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    
    # Image parameters
    size = 256
    noise_levels = [0.05, 0.10, 0.15, 0.20]
    seed = 42
    
    # Test image generators
    generators = {
        'simple': create_simple_image,
        'edges': create_edges_image,
        'textures': create_textures_image,
        'gradients': create_gradients_image,
    }
    
    print(f"\nImage size: {size}x{size}")
    print(f"Noise levels: {noise_levels}")
    print(f"Random seed: {seed}")
    print(f"\nGenerating {len(generators)} test images...")
    
    # Generate clean images
    for name, generator in generators.items():
        print(f"\n  Creating {name} image...")
        clean_img = generator(size)
        clean_path = clean_dir / f"{name}.png"
        save_image_4bit(clean_img, clean_path)
        print(f"    Saved: {clean_path}")
        
        # Generate noisy versions
        for sigma in noise_levels:
            noisy_img = add_gaussian_noise(clean_img, sigma, seed=seed)
            noisy_path = noisy_dir / f"{name}_sigma{sigma:.2f}.png"
            save_image_4bit(noisy_img, noisy_path)
            print(f"    Saved: {noisy_path}")
    
    print(f"\n{'='*70}")
    print("Test images generated successfully!")
    print(f"{'='*70}")
    print(f"\nClean images: {clean_dir}")
    print(f"Noisy images: {noisy_dir}")
    print(f"\nTotal files:")
    print(f"  Clean: {len(list(clean_dir.glob('*.png')))}")
    print(f"  Noisy: {len(list(noisy_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
