"""JAX-based bilateral filter implementation for GPU-accelerated denoising."""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import numpy as np


def bilateral_filter_kernel(
    image: jnp.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> jnp.ndarray:
    """
    Bilateral filter implementation using JAX.
    
    This is a simplified vectorized version optimized for GPU.
    
    Args:
        image: Input image, shape (H, W), values in [0, 1]
        d: Diameter of neighborhood (should be odd)
        sigma_color: Filter sigma for color/intensity domain (already normalized to [0, 1])
        sigma_space: Filter sigma for spatial domain (already normalized to [0, 1])
    
    Returns:
        Filtered image, same shape as input
    """
    # Note: sigma_color and sigma_space should be normalized before calling this function
    
    H, W = image.shape
    pad = d // 2
    
    # Pad image
    padded = jnp.pad(image, ((pad, pad), (pad, pad)), mode='edge')
    
    # Create coordinate grids
    y_coords, x_coords = jnp.meshgrid(
        jnp.arange(H, dtype=jnp.float32),
        jnp.arange(W, dtype=jnp.float32),
        indexing='ij'
    )
    
    # Initialize output
    output = jnp.zeros_like(image)
    weight_sum = jnp.zeros_like(image)
    
    # Vectorized bilateral filter
    # For each pixel, compute weighted average of neighbors
    # JAX will unroll these static loops during JIT compilation
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            # Spatial weight
            spatial_dist_sq = (dy * dy + dx * dx) / (sigma_space * sigma_space + 1e-8)
            spatial_weight = jnp.exp(-0.5 * spatial_dist_sq)
            
            # Get neighbor values from padded image (using proper indexing)
            # Since we padded, we can directly index with offset
            # Convert to int for indexing
            y_idx = (y_coords + dy + pad).astype(jnp.int32)
            x_idx = (x_coords + dx + pad).astype(jnp.int32)
            neighbor_values = padded[y_idx, x_idx]
            
            # Color/intensity weight
            color_diff = neighbor_values - image
            color_dist_sq = (color_diff * color_diff) / (sigma_color * sigma_color + 1e-8)
            color_weight = jnp.exp(-0.5 * color_dist_sq)
            
            # Combined weight
            weight = spatial_weight * color_weight
            
            # Accumulate weighted values
            output = output + neighbor_values * weight
            weight_sum = weight_sum + weight
    
    # Normalize
    output = output / (weight_sum + 1e-8)
    
    return jnp.clip(output, 0.0, 1.0)


def bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Apply bilateral filter to image.
    
    Args:
        image: Input image, shape (H, W), values in [0, 1]
        d: Diameter of neighborhood (should be odd)
        sigma_color: Filter sigma for color/intensity domain
        sigma_space: Filter sigma for spatial domain
    
    Returns:
        Filtered image as numpy array
    """
    # Normalize sigma values to [0, 1] range if needed (before JIT)
    if sigma_color > 1.0:
        sigma_color = sigma_color / 255.0
    if sigma_space > 1.0:
        sigma_space = sigma_space / 255.0
    
    # Convert to JAX array
    jax_image = jnp.array(image)
    
    # Apply filter with JIT (d is static argument - it's the 2nd positional arg, index 1)
    # Use static_argnums to specify d is static
    filtered = jax.jit(bilateral_filter_kernel, static_argnums=(1,))(jax_image, d, sigma_color, sigma_space)
    
    # Convert back to numpy
    return np.array(filtered)


class BilateralFilterJAX:
    """JAX-based bilateral filter with batch processing support."""
    
    def __init__(
        self,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
    ):
        """
        Initialize bilateral filter.
        
        Args:
            d: Diameter of neighborhood (should be odd)
            sigma_color: Filter sigma for color/intensity domain
            sigma_space: Filter sigma for spatial domain
        """
        self.d = d
        
        # Normalize sigma values to [0, 1] range if needed (before JIT)
        if sigma_color > 1.0:
            sigma_color = sigma_color / 255.0
        if sigma_space > 1.0:
            sigma_space = sigma_space / 255.0
        
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        
        # JIT-compiled filter function (with normalized sigmas)
        # Create a closure that captures d, sigma_color, sigma_space
        # We need to JIT-compile with d as static argument
        _d = d  # Capture as local variable
        _sigma_color = sigma_color
        _sigma_space = sigma_space
        
        def make_filter_fn(img):
            # Call bilateral_filter_kernel with d as static (index 1)
            return jax.jit(bilateral_filter_kernel, static_argnums=(1,))(
                img, _d, _sigma_color, _sigma_space
            )
        
        # JIT compile the wrapper function
        self._filter_fn = jax.jit(make_filter_fn)
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Denoise a single image.
        
        Args:
            image: Input image, shape (H, W), values in [0, 1]
        
        Returns:
            Denoised image as numpy array
        """
        jax_image = jnp.array(image)
        filtered = self._filter_fn(jax_image)
        return np.array(filtered)
    
    def denoise_batch(self, images: list) -> list:
        """
        Denoise a batch of images.
        
        Args:
            images: List of input images, each shape (H, W)
        
        Returns:
            List of denoised images
        """
        return [self.denoise(img) for img in images]

