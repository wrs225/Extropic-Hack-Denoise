"""Quality metrics and energy efficiency calculations."""

import numpy as np
from typing import Tuple, Dict, Any


def calculate_psnr(image1: np.ndarray, image2: np.ndarray, max_value: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        image1: First image (reference)
        image2: Second image (to compare)
        max_value: Maximum pixel value (default 1.0 for normalized images)
    
    Returns:
        PSNR in dB
    """
    mse = np.mean((image1 - image2) ** 2)
    
    if mse == 0:
        return float('inf')  # Images are identical
    
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (SSIM).
    
    Simplified SSIM calculation for grayscale images.
    
    Args:
        image1: First image (reference)
        image2: Second image (to compare)
    
    Returns:
        SSIM value in [0, 1]
    """
    # Constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Calculate means
    mu1 = np.mean(image1)
    mu2 = np.mean(image2)
    
    # Calculate variances and covariance
    sigma1_sq = np.var(image1)
    sigma2_sq = np.var(image2)
    sigma12 = np.mean((image1 - mu1) * (image2 - mu2))
    
    # Calculate SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = numerator / (denominator + 1e-8)
    return float(ssim)


def calculate_improvement_metrics(
    original: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate improvement metrics comparing noisy vs denoised.
    
    Args:
        original: Original clean image
        noisy: Noisy input image
        denoised: Denoised output image
    
    Returns:
        Dictionary with improvement metrics
    """
    # Calculate PSNR for noisy and denoised
    psnr_noisy = calculate_psnr(original, noisy)
    psnr_denoised = calculate_psnr(original, denoised)
    psnr_improvement = psnr_denoised - psnr_noisy
    
    # Calculate SSIM for noisy and denoised
    ssim_noisy = calculate_ssim(original, noisy)
    ssim_denoised = calculate_ssim(original, denoised)
    ssim_improvement = ssim_denoised - ssim_noisy
    
    # Calculate MSE reduction
    mse_noisy = np.mean((original - noisy) ** 2)
    mse_denoised = np.mean((original - denoised) ** 2)
    if mse_noisy > 0:
        mse_reduction_pct = ((mse_noisy - mse_denoised) / mse_noisy) * 100.0
    else:
        mse_reduction_pct = 0.0
    
    return {
        'psnr_noisy': float(psnr_noisy),
        'psnr_denoised': float(psnr_denoised),
        'psnr_improvement': float(psnr_improvement),
        'ssim_noisy': float(ssim_noisy),
        'ssim_denoised': float(ssim_denoised),
        'ssim_improvement': float(ssim_improvement),
        'mse_reduction_pct': float(mse_reduction_pct),
    }


def calculate_efficiency_metrics(
    psnr: float,
    energy_j: float,
    image_shape: Tuple[int, ...],
) -> Dict[str, float]:
    """
    Calculate energy efficiency metrics.
    
    Args:
        psnr: PSNR value in dB
        energy_j: Energy consumption in Joules
        image_shape: Shape of processed image (H, W) or (H, W, C)
    
    Returns:
        Dictionary with efficiency metrics
    """
    # Calculate number of pixels
    if len(image_shape) >= 2:
        n_pixels = int(np.prod(image_shape[:2]))
    else:
        n_pixels = int(np.prod(image_shape))
    
    # Energy per pixel (nanojoules)
    energy_per_pixel_nj = (energy_j / n_pixels) * 1e9
    
    # PSNR per Joule
    if energy_j > 0:
        psnr_per_joule = psnr / energy_j
    else:
        psnr_per_joule = float('inf')
    
    return {
        'energy_per_pixel_nj': float(energy_per_pixel_nj),
        'psnr_per_joule': float(psnr_per_joule),
        'n_pixels': n_pixels,
    }


def export_results_to_json(
    results: Dict[str, Any],
    filepath: str,
) -> None:
    """
    Export benchmark results to JSON file.
    
    Args:
        results: Dictionary with benchmark results
        filepath: Output file path
    """
    import json
    from datetime import datetime
    
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results = convert_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

