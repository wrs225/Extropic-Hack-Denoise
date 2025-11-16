"""Reproducible noise generation for testing."""

import numpy as np
from typing import Optional, Tuple


class NoiseGenerator:
    """Generator for reproducible noise patterns."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize noise generator with optional seed.
        
        Args:
            seed: Random seed for reproducibility. If None, uses current time.
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def gaussian(
        self,
        image: np.ndarray,
        sigma: float,
        relative: bool = True,
    ) -> np.ndarray:
        """
        Add Gaussian noise to image.
        
        Args:
            image: Input image array
            sigma: Noise standard deviation. If relative=True, this is a fraction of value.
            relative: If True, sigma is interpreted as fraction (e.g., 0.15 = 15%)
        
        Returns:
            Noisy image
        """
        if relative:
            # sigma is percentage of value
            noise_std = image * sigma
        else:
            noise_std = sigma
        
        # Generate noise
        noise = np.random.normal(0.0, noise_std, size=image.shape).astype(np.float32)
        
        # Add noise
        noisy = image + noise
        
        # Clamp to valid range
        noisy = np.clip(noisy, 0.0, 1.0)
        
        return noisy
    
    def salt_and_pepper(
        self,
        image: np.ndarray,
        salt_prob: float = 0.05,
        pepper_prob: float = 0.05,
    ) -> np.ndarray:
        """
        Add salt and pepper noise.
        
        Args:
            image: Input image array
            salt_prob: Probability of salt (white) noise
            pepper_prob: Probability of pepper (black) noise
        
        Returns:
            Noisy image
        """
        noisy = image.copy()
        
        # Generate random values
        random_values = np.random.random(image.shape)
        
        # Add salt (white)
        salt_mask = random_values < salt_prob
        noisy[salt_mask] = 1.0
        
        # Add pepper (black)
        pepper_mask = (random_values >= salt_prob) & (random_values < salt_prob + pepper_prob)
        noisy[pepper_mask] = 0.0
        
        return noisy


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float = 0.15,
    relative: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function to add Gaussian noise.
    
    Args:
        image: Input image array
        sigma: Noise standard deviation (default 15% = 0.15)
        relative: If True, sigma is fraction of value
        seed: Random seed for reproducibility
    
    Returns:
        Noisy image
    """
    generator = NoiseGenerator(seed=seed)
    return generator.gaussian(image, sigma, relative)

