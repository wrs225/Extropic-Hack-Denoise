#!/usr/bin/env python3
"""
Unified comparison script for denoising algorithms.

Compares all available denoisers on identical test images with identical noise,
ensuring fair performance and quality comparisons.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

# Add parent directory for examples
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import DenoisingPipeline, ImageDataStore, ComparisonVisualizer
from denoisers import IsingMultiLayerDenoiser


def load_4bit_image(path: Path) -> np.ndarray:
    """Load 4-bit image and convert to [0, 1] float32."""
    img = Image.open(path)
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convert to numpy
    img_array = np.array(img, dtype=np.float32)
    
    # Convert from 8-bit (0-255) to 4-bit (0-15) then normalize
    img_4bit = (img_array / 255.0 * 15.0).astype(np.uint8)
    
    # Normalize to [0, 1]
    return (img_4bit.astype(np.float32) / 15.0)


def prepare_test_images():
    """Generate test images if they don't exist."""
    from prepare_test_images import main as prepare_main
    
    print("Preparing test images...")
    prepare_main()


def run_comparison(
    image_name: str = None,
    sigma: float = None,
    custom_image_path: Path = None,
    output_dir: Path = None,
):
    """
    Run comparison of all denoisers.
    
    Args:
        image_name: Name of test image (simple, edges, textures, gradients)
        sigma: Noise level (0.05, 0.10, 0.15, 0.20)
        custom_image_path: Path to custom image (overrides image_name)
        output_dir: Output directory for results
    """
    print("="*70)
    print("Unified Denoiser Comparison")
    print("="*70)
    
    # Initialize pipeline
    pipeline = DenoisingPipeline(gpu_id=0)
    datastore = ImageDataStore()
    visualizer = ComparisonVisualizer()
    
    if output_dir is None:
        output_dir = datastore.output_dir
    
    # Load images
    if custom_image_path:
        print(f"\nLoading custom image: {custom_image_path}")
        clean = datastore.load_image(custom_image_path)
        # Add noise with specified sigma
        if sigma is None:
            sigma = 0.15
        np.random.seed(42)
        noise = np.random.normal(0, sigma, clean.shape).astype(np.float32)
        noisy = np.clip(clean + noise, 0.0, 1.0)
        image_name = custom_image_path.stem
    else:
        # Load from common test set
        if image_name is None:
            image_name = "simple"
        if sigma is None:
            sigma = 0.15
        
        print(f"\nLoading test image: {image_name} (sigma={sigma})")
        test_dir = Path("data/common_test")
        clean_path = test_dir / "clean" / f"{image_name}.png"
        noisy_path = test_dir / "noisy" / f"{image_name}_sigma{sigma:.2f}.png"
        
        if not clean_path.exists() or not noisy_path.exists():
            print(f"Test images not found. Generating...")
            prepare_test_images()
        
        clean = load_4bit_image(clean_path)
        noisy = load_4bit_image(noisy_path)
    
    print(f"  Clean image shape: {clean.shape}, range: [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"  Noisy image shape: {noisy.shape}, range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    
    # Initialize denoisers
    print("\n" + "-"*70)
    print("Initializing denoisers...")
    print("-"*70)
    
    denoisers = {}
    
    # Bilateral Filter (if available)
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src-temp"))
        from s2a_baseline.bilateral_filter import BilateralFilterJAX
        bilateral = BilateralFilterJAX(d=11, sigma_color=30.0, sigma_space=100.0)
        denoisers["Bilateral Filter"] = bilateral
        print("  ✓ Bilateral Filter initialized")
    except ImportError:
        print("  ⚠ Bilateral Filter not available (skipping)")
    
    # ISling Denoiser
    try:
        ising = IsingMultiLayerDenoiser(
            h=clean.shape[0],
            w=clean.shape[1],
            num_bits=4,
            n_warmup=20,
            n_samples=10,
            steps_per_sample=5,
            alpha_coef=0.5,
            beta_coef=1.0,
            temperature=1.0,
        )
        denoisers["Ising Multi-Layer"] = ising
        print("  ✓ Ising Multi-Layer initialized")
    except Exception as e:
        print(f"  ⚠ Ising Multi-Layer not available: {e}")
    
    if len(denoisers) == 0:
        print("\n❌ No denoisers available!")
        return
    
    # Run denoisers
    print("\n" + "-"*70)
    print("Running denoisers...")
    print("-"*70)
    
    all_results = {}
    summary_data = {
        'image': str(custom_image_path) if custom_image_path else f"{image_name}",
        'noise_sigma': float(sigma),
        'noise_seed': 42,
        'timestamp': datetime.now().isoformat(),
        'noisy': {},
        'algorithms': {},
    }
    
    # Calculate noisy metrics
    from pipeline.metrics import calculate_all_metrics
    noisy_metrics = calculate_all_metrics(clean, clean, noisy)  # Compare noisy to clean
    summary_data['noisy'] = {
        'psnr': noisy_metrics['psnr_noisy'],
        'ssim': noisy_metrics['ssim_noisy'],
    }
    
    for name, denoiser in denoisers.items():
        print(f"\n  Processing with {name}...")
        
        try:
            # Process through pipeline
            results = pipeline.process(
                denoiser=denoiser,
                clean_image=clean,
                noisy_image=noisy,
                metadata={
                    'algorithm': name,
                    'image': image_name,
                    'sigma': sigma,
                }
            )
            
            # Save individual results
            exp_name = f"{image_name}_sigma{sigma:.2f}_{name.lower().replace(' ', '_')}"
            exp_dir = datastore.save_results(results, experiment_name=exp_name)
            
            # Create comparison visualization
            visualizer.create_comparison(
                clean=results['clean'],
                noisy=results['noisy'],
                denoised=results['denoised'],
                output_path=exp_dir / "comparison.png",
                metrics=results['metrics'],
                algorithm_name=name,
            )
            
            # Store for summary
            all_results[name] = results
            summary_data['algorithms'][name] = {
                'psnr': float(results['metrics']['psnr_denoised']),
                'psnr_improvement': float(results['metrics']['psnr_improvement']),
                'ssim': float(results['metrics']['ssim_denoised']),
                'ssim_improvement': float(results['metrics']['ssim_improvement']),
                'mse_reduction_pct': float(results['metrics']['mse_reduction_pct']),
                'energy_j': float(results['power']['mean_energy_j']),
                'time_ms': float(results['performance']['mean_time_ms']),
                'metadata': results['metadata'],
            }
            
            print(f"    ✓ PSNR: {results['metrics']['psnr_denoised']:.2f} dB "
                  f"(+{results['metrics']['psnr_improvement']:.2f} dB)")
            print(f"    ✓ Energy: {results['power']['mean_energy_j']:.3f} J")
            print(f"    ✓ Time: {results['performance']['mean_time_ms']:.2f} ms")
            print(f"    ✓ Results saved to: {exp_dir}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Save comparison summary
    summary_path = output_dir / f"{image_name}_sigma{sigma:.2f}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print("\n" + "="*70)
    print("Comparison Complete!")
    print("="*70)
    print(f"\nSummary saved to: {summary_path}")
    print("\nResults Summary:")
    print("-"*70)
    for name, data in summary_data['algorithms'].items():
        print(f"\n{name}:")
        print(f"  PSNR: {data['psnr']:.2f} dB (improvement: {data['psnr_improvement']:+.2f} dB)")
        print(f"  SSIM: {data['ssim']:.4f} (improvement: {data['ssim_improvement']:+.4f})")
        print(f"  Energy: {data['energy_j']:.3f} J")
        print(f"  Time: {data['time_ms']:.2f} ms")
    print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified comparison of denoising algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare test images first
  python main_comparison.py --prepare
  
  # Run comparison on default test image
  python main_comparison.py
  
  # Run on specific test image and noise level
  python main_comparison.py --image simple --sigma 0.15
  
  # Run on custom image
  python main_comparison.py --image path/to/image.png --sigma 0.15
        """
    )
    parser.add_argument(
        '--prepare',
        action='store_true',
        help='Generate test images'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Test image name (simple, edges, textures, gradients) or path to custom image'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.15,
        help='Noise level (default: 0.15)'
    )
    
    args = parser.parse_args()
    
    if args.prepare:
        prepare_test_images()
    else:
        # Determine if custom image or test image name
        image_path = Path(args.image) if args.image else None
        if image_path and image_path.exists():
            # Custom image
            run_comparison(
                custom_image_path=image_path,
                sigma=args.sigma,
            )
        else:
            # Test image name
            run_comparison(
                image_name=args.image or "simple",
                sigma=args.sigma,
            )


if __name__ == '__main__':
    main()
