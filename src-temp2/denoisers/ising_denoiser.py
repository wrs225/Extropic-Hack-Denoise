"""ISling Multi-Layer Denoiser wrapper for pipeline integration."""

import numpy as np
import sys
from pathlib import Path

# Add examples directory to path
examples_path = Path(__file__).parent.parent.parent / "examples"
sys.path.insert(0, str(examples_path))

from denoise import MultiLayerDenoiser


class IsingMultiLayerDenoiser:
    """
    Wrapper for ISling MultiLayerDenoiser to work with pipeline interface.
    
    Converts between pipeline format ([0, 1] float32) and ISling format ([0, 15] uint8).
    Handles JIT compilation and warmup automatically.
    """
    
    def __init__(
        self,
        h: int = 256,
        w: int = 256,
        num_bits: int = 4,
        n_warmup: int = 20,
        n_samples: int = 10,
        steps_per_sample: int = 5,
        alpha_coef: float = 0.5,
        beta_coef: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        Initialize ISling denoiser.
        
        Args:
            h: Image height
            w: Image width
            num_bits: Bit depth (default 4 for 4-bit images)
            n_warmup: Number of warmup samples for MCMC
            n_samples: Number of samples to average
            steps_per_sample: MCMC steps per sample
            alpha_coef: Fidelity coefficient (higher = stay closer to noisy input)
            beta_coef: Smoothness coefficient (higher = more smoothing)
            temperature: Sampling temperature
        """
        self.h = h
        self.w = w
        self.num_bits = num_bits
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.steps_per_sample = steps_per_sample
        self.alpha_coef = alpha_coef
        self.beta_coef = beta_coef
        self.temperature = temperature
        
        # Initialize the underlying denoiser
        self.denoiser = MultiLayerDenoiser(
            h=h,
            w=w,
            num_bits=num_bits,
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=steps_per_sample,
            alpha_coef=alpha_coef,
            beta_coef=beta_coef,
            temperature=temperature,
        )
        
        # Track if we've done initial warmup
        self._initialized = False
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Denoise a single image.
        
        Args:
            image: Input noisy image, shape (H, W), values in [0, 1] (float32)
        
        Returns:
            Denoised image, same shape, values in [0, 1] (float32)
        """
        # Ensure image is in correct format
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Clip to valid range
        image = np.clip(image, 0.0, 1.0)
        
        # Convert from pipeline format [0, 1] to ISling format [0, 15]
        img_4bit = (image * 15.0).astype(np.uint8)
        
        # Do initial warmup on first call
        warm_up = not self._initialized
        
        # Denoise using ISling
        denoised_4bit, _, _ = self.denoiser.denoise_image(
            img_4bit,
            key=None,  # Will use default seed
            warm_up=warm_up,
            verbose=False,
        )
        
        # Mark as initialized after first call
        if not self._initialized:
            self._initialized = True
        
        # Convert back from ISling format [0, 15] to pipeline format [0, 1]
        denoised = denoised_4bit.astype(np.float32) / 15.0
        
        # Ensure output is in valid range
        denoised = np.clip(denoised, 0.0, 1.0)
        
        return denoised
    
    def denoise_batch(self, images: list) -> list:
        """
        Denoise a batch of images (optional, can default to sequential).
        
        Args:
            images: List of input noisy images
        
        Returns:
            List of denoised images
        """
        return [self.denoise(img) for img in images]
