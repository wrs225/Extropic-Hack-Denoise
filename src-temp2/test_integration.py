#!/usr/bin/env python3
"""
Integration test for the denoising pipeline.

Quick verification that everything works:
- Test image creation
- Pipeline initialization
- ISling denoiser loading
- Full processing pipeline
- Result saving and visualization
"""

import sys
from pathlib import Path
import numpy as np

print("="*70)
print("Integration Test: Denoising Pipeline")
print("="*70)

# Test 1: Test image creation
print("\n[1/5] Testing image creation...")
try:
    from prepare_test_images import create_simple_image, add_gaussian_noise
    
    clean = create_simple_image(256)
    noisy = add_gaussian_noise(clean, sigma=0.15, seed=42)
    
    assert clean.shape == (256, 256), f"Expected (256, 256), got {clean.shape}"
    assert noisy.shape == (256, 256), f"Expected (256, 256), got {noisy.shape}"
    assert clean.min() >= 0.0 and clean.max() <= 1.0, "Clean image out of range"
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0, "Noisy image out of range"
    
    print("  ✓ Test images created successfully")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Pipeline initialization
print("\n[2/5] Testing pipeline initialization...")
try:
    from pipeline import DenoisingPipeline
    
    pipeline = DenoisingPipeline(gpu_id=0)
    print("  ✓ Pipeline initialized successfully")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Data store
print("\n[3/5] Testing data store...")
try:
    from pipeline import ImageDataStore
    
    datastore = ImageDataStore()
    print(f"  ✓ Data store initialized: {datastore.base_dir}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 4: ISling denoiser loading
print("\n[4/5] Testing ISling denoiser loading...")
try:
    from denoisers import IsingMultiLayerDenoiser
    
    denoiser = IsingMultiLayerDenoiser(
        h=256,
        w=256,
        num_bits=4,
        n_warmup=5,  # Reduced for faster testing
        n_samples=5,  # Reduced for faster testing
        steps_per_sample=5,
    )
    print("  ✓ ISling denoiser loaded successfully")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    print("  Note: This requires examples/denoise/ to be available")
    sys.exit(1)

# Test 5: Full processing pipeline
print("\n[5/5] Testing full processing pipeline...")
try:
    from pipeline import ComparisonVisualizer
    
    # Create test images
    clean = create_simple_image(256)
    noisy = add_gaussian_noise(clean, sigma=0.15, seed=42)
    
    # Process through pipeline
    print("  Processing (this may take 10-20 seconds for JIT compilation)...")
    results = pipeline.process(
        denoiser=denoiser,
        clean_image=clean,
        noisy_image=noisy,
        metadata={
            'algorithm': 'Ising Multi-Layer',
            'test': True,
        }
    )
    
    # Verify results
    assert 'denoised' in results, "Missing denoised image"
    assert 'metrics' in results, "Missing metrics"
    assert 'power' in results, "Missing power data"
    assert 'performance' in results, "Missing performance data"
    
    assert results['denoised'].shape == (256, 256), "Wrong denoised shape"
    assert results['metrics']['psnr_denoised'] > 0, "Invalid PSNR"
    
    print(f"  ✓ Processing complete")
    print(f"    PSNR improvement: {results['metrics']['psnr_improvement']:.2f} dB")
    print(f"    Energy: {results['power']['mean_energy_j']:.3f} J")
    print(f"    Time: {results['performance']['mean_time_ms']:.2f} ms")
    
    # Save results
    exp_dir = datastore.save_results(results, experiment_name="integration_test")
    print(f"  ✓ Results saved to: {exp_dir}")
    
    # Create comparison
    visualizer = ComparisonVisualizer()
    visualizer.create_comparison(
        clean=results['clean'],
        noisy=results['noisy'],
        denoised=results['denoised'],
        output_path=exp_dir / "comparison.png",
        metrics=results['metrics'],
        algorithm_name="Ising Multi-Layer",
    )
    print(f"  ✓ Comparison image created")
    
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("All integration tests passed! ✓")
print("="*70)
print("\nThe pipeline is ready to use.")
print("Next steps:")
print("  1. Generate test images: python prepare_test_images.py")
print("  2. Run comparison: python main_comparison.py")
print()

