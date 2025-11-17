"""
Parameter Sweep for Cameraman Image with Different Salt & Pepper Noise Levels

Tests salt and pepper noise levels: 0.05, 0.10, and 0.15
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils import add_salt_pepper_noise, compute_metrics
from denoiser import MultiLayerDenoiser


def load_cameraman_image(cameraman_path):
    """Load and convert cameraman image to 4-bit greyscale."""
    img = Image.open(cameraman_path)

    # Convert to greyscale
    img_grey = img.convert('L')

    # Resize to 256x256 for faster processing
    img_grey = img_grey.resize((256, 256), Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img_grey)

    # Convert to 4-bit (0-15)
    img_4bit = (img_array / 255.0 * 15).astype(np.uint8)

    return img_4bit


def run_parameter_sweep(noisy_img, original_img, noise_level, output_dir):
    """Run parameter sweep for a specific noise level."""

    os.makedirs(output_dir, exist_ok=True)

    # Define parameter ranges
    alpha_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    beta_values = [0.3, 0.5, 0.7, 1.0, 1.5]

    results = []
    h, w = noisy_img.shape

    print(f"\n{'='*70}")
    print(f"PARAMETER SWEEP - CAMERAMAN IMAGE - SALT & PEPPER NOISE {noise_level}")
    print(f"{'='*70}")
    print(f"Testing {len(alpha_values)} × {len(beta_values)} = {len(alpha_values) * len(beta_values)} parameter combinations")
    print(f"Alpha (fidelity) values: {alpha_values}")
    print(f"Beta (smoothness) values: {beta_values}")
    print(f"{'='*70}")

    total_combinations = len(alpha_values) * len(beta_values)
    current = 0

    # JAX key for reproducibility
    key = jax.random.PRNGKey(42)

    # Initialize denoiser ONCE
    print("\nInitializing denoiser (building graph + JIT compilation)...")
    denoiser = MultiLayerDenoiser(
        h=h, w=w,
        n_warmup=20,
        n_samples=10,
        steps_per_sample=200,
        alpha_coef=0.5,
        beta_coef=1.0,
        temperature=1.0,
        verbose=False
    )
    print("Denoiser initialized!\n")

    # Run sweep
    for alpha in alpha_values:
        for beta in beta_values:
            current += 1
            print(f"[{current}/{total_combinations}] α={alpha:.1f}, β={beta:.1f}...", end='', flush=True)

            # Denoise with these parameters
            denoised_img, individual_imgs, denoise_time = denoiser.denoise_image(
                noisy_img,
                key=key,
                warm_up=(current == 1),
                verbose=False,
                alpha_coef=alpha,
                beta_coef=beta,
                temperature=1.0
            )

            # Compute metrics
            metrics = compute_metrics(original_img, noisy_img, denoised_img)

            # Store results
            result = {
                'alpha': float(alpha),
                'beta': float(beta),
                'psnr_noisy': float(metrics['psnr_noisy']),
                'psnr_denoised': float(metrics['psnr_denoised']),
                'psnr_improvement': float(metrics['psnr_improvement']),
                'denoise_time': float(denoise_time),
            }
            results.append(result)

            print(f" → PSNR: {metrics['psnr_denoised']:.2f} dB (+{metrics['psnr_improvement']:.2f} dB)")

            # Save this result's image
            img_filename = f"{output_dir}/denoised_a{alpha:.1f}_b{beta:.1f}.png"
            Image.fromarray((denoised_img * 17).astype(np.uint8)).save(img_filename)

    # Find best parameters
    best_result = max(results, key=lambda x: x['psnr_denoised'])

    print(f"\n{'='*70}")
    print("SWEEP COMPLETE!")
    print(f"{'='*70}")
    print(f"Best parameters for noise level {noise_level}:")
    print(f"  alpha_coef: {best_result['alpha']:.1f}")
    print(f"  beta_coef: {best_result['beta']:.1f}")
    print(f"  PSNR improvement: {best_result['psnr_improvement']:.2f} dB")
    print(f"  Final PSNR: {best_result['psnr_denoised']:.2f} dB")
    print(f"{'='*70}")

    # Save results to JSON
    results_file = f"{output_dir}/sweep_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'noise_level': noise_level,
            'results': results,
            'best': best_result,
            'alpha_values': alpha_values,
            'beta_values': beta_values
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Create heatmap
    create_heatmap(results, alpha_values, beta_values, noise_level, output_dir)

    return results, best_result


def create_heatmap(results, alpha_values, beta_values, noise_level, output_dir):
    """Create heatmap visualization."""

    psnr_matrix = np.zeros((len(beta_values), len(alpha_values)))
    for result in results:
        alpha_idx = alpha_values.index(result['alpha'])
        beta_idx = beta_values.index(result['beta'])
        psnr_matrix[beta_idx, alpha_idx] = result['psnr_improvement']

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    im = ax.imshow(psnr_matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(alpha_values)))
    ax.set_yticks(range(len(beta_values)))
    ax.set_xticklabels([f"{a:.1f}" for a in alpha_values])
    ax.set_yticklabels([f"{b:.1f}" for b in beta_values])
    ax.set_xlabel('Alpha (Fidelity Coefficient)', fontsize=12)
    ax.set_ylabel('Beta (Smoothness Coefficient)', fontsize=12)
    ax.set_title(f'PSNR Improvement (dB) - Cameraman - Salt&Pepper Noise={noise_level}', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(beta_values)):
        for j in range(len(alpha_values)):
            text = ax.text(j, i, f'{psnr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax, label='PSNR Improvement (dB)')
    plt.tight_layout()

    heatmap_file = f"{output_dir}/heatmap_noise{noise_level}.png"
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to: {heatmap_file}")
    plt.close()


def main():
    """Run parameter sweeps for cameraman image with different salt & pepper noise levels."""

    # Load cameraman image
    print("Loading cameraman image...")
    cameraman_path = "/home/will/Extropic-Hack/data/cameraman.png"
    original_img = load_cameraman_image(cameraman_path)
    print(f"Loaded cameraman image: {original_img.shape}, range: [{original_img.min()}, {original_img.max()}]")

    # Save original
    os.makedirs("build/cameraman_sweeps", exist_ok=True)
    Image.fromarray((original_img * 17).astype(np.uint8)).save("build/cameraman_sweeps/original.png")

    # Test different noise levels
    noise_levels = [0.05, 0.10, 0.15]
    all_results = {}

    for noise_level in noise_levels:
        print(f"\n\n{'#'*70}")
        print(f"# TESTING NOISE LEVEL: {noise_level}")
        print(f"{'#'*70}")

        # Add salt and pepper noise
        np.random.seed(42)
        noisy_img = add_salt_pepper_noise(original_img, noise_level=noise_level)

        # Save noisy image
        noisy_path = f"build/cameraman_sweeps/noisy_{noise_level}.png"
        Image.fromarray((noisy_img * 17).astype(np.uint8)).save(noisy_path)
        print(f"Noisy image saved: {noisy_path}")

        # Run sweep
        output_dir = f"build/cameraman_sweeps/noise_{noise_level}"
        results, best_result = run_parameter_sweep(noisy_img, original_img, noise_level, output_dir)

        all_results[noise_level] = {
            'results': results,
            'best': best_result
        }

    # Summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY - CAMERAMAN IMAGE")
    print(f"{'='*70}")
    for noise_level in noise_levels:
        best = all_results[noise_level]['best']
        print(f"\nNoise level {noise_level}:")
        print(f"  Best α={best['alpha']:.1f}, β={best['beta']:.1f}")
        print(f"  PSNR: {best['psnr_denoised']:.2f} dB (+{best['psnr_improvement']:.2f} dB)")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
