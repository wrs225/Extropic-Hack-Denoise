# Agnostic Denoising Pipeline

A clean, modular pipeline infrastructure for comparing image denoising algorithms with GPU power monitoring. Includes ISling Multi-Layer Denoiser integration for fair comparison.

## Quick Start

```bash
# Step 1: Verify integration works
python test_integration.py

# Step 2: Generate standardized test images
python prepare_test_images.py

# Step 3: Compare all denoisers on test images
python main_comparison.py

# Or run basic examples
python main.py
```

## Directory Structure

```
src-temp2/
├── denoisers/              # Denoiser implementations
│   ├── ising_denoiser.py  # ISling wrapper
│   └── __init__.py
├── pipeline/               # Pipeline infrastructure
│   ├── core.py            # DenoisingPipeline
│   ├── datastore.py       # ImageDataStore
│   ├── benchmark.py       # GPU monitoring
│   ├── metrics.py         # Quality metrics
│   └── comparison.py      # Visualization
├── data/
│   ├── common_test/       # Standardized test images
│   │   ├── clean/         # Clean reference images
│   │   └── noisy/         # Noisy test images
│   ├── input/             # User input images
│   └── output/            # Results saved here
├── prepare_test_images.py # Generate test dataset
├── main_comparison.py     # Unified comparison script
├── test_integration.py    # Integration test
├── main.py                # Basic examples
└── README.md              # This file
```

## Usage

### 1. Integration Test

Verify everything works:
```bash
python test_integration.py
```

### 2. Prepare Test Images

Generate standardized test images for fair comparison:
```bash
python prepare_test_images.py
```

This creates:
- `data/common_test/clean/` - 4 test image types (simple, edges, textures, gradients)
- `data/common_test/noisy/` - Noisy versions with σ = [0.05, 0.10, 0.15, 0.20]

### 3. Compare Denoisers

Run comparison on all test images:
```bash
python main_comparison.py
```

Or on a specific image:
```bash
python main_comparison.py --image data/common_test/clean/simple.png --sigma 0.15
```

Or on a custom image:
```bash
python main_comparison.py --image path/to/image.png --sigma 0.15
```

### 4. Check Results

Results are saved to `data/output/`:
- Individual results: `{image}_sigma{level}_{algorithm}_*/`
  - `comparison.png` - Visual comparison
  - `metrics.json` - Quality metrics
  - `results.json` - Full results with GPU power
- Comparison summary: `{image}_sigma{level}_summary.json`

## Available Denoisers

### ISling Multi-Layer Denoiser

4-bit denoising using thermodynamic Ising machine:
```python
from denoisers import IsingMultiLayerDenoiser

denoiser = IsingMultiLayerDenoiser(
    h=256, w=256, num_bits=4,
    n_warmup=20, n_samples=10, steps_per_sample=5,
    alpha_coef=0.5, beta_coef=1.0, temperature=1.0
)
```

### Bilateral Filter (if available)

JAX-based bilateral filter:
```python
from s2a_baseline.bilateral_filter import BilateralFilterJAX

denoiser = BilateralFilterJAX(d=11, sigma_color=30.0, sigma_space=100.0)
```

## Adding New Denoisers

1. Create denoiser class in `denoisers/`:
```python
# denoisers/my_denoiser.py
class MyDenoiser:
    def denoise(self, image: np.ndarray) -> np.ndarray:
        # Input: (H, W), values [0, 1]
        # Output: (H, W), values [0, 1]
        return denoised_image
```

2. Export in `denoisers/__init__.py`:
```python
from .my_denoiser import MyDenoiser
__all__ = [..., "MyDenoiser"]
```

3. Add to comparison in `main_comparison.py`:
```python
from denoisers import MyDenoiser

denoiser = MyDenoiser()
results = pipeline.process(denoiser, clean, noisy, metadata)
```

That's it! Your denoiser will be compared fairly against all others.

## Fair Comparison Guarantees

1. **Same Input Images**: All denoisers tested on identical images from `data/common_test/`
2. **Same Noise**: Fixed seed (42), same σ values
3. **Same Metrics**: Identical PSNR/SSIM calculations
4. **Same Hardware**: Same GPU, same monitoring setup
5. **Full Metadata**: All parameters saved for reproducibility

## Features

- ✅ **GPU-First**: Automatic JAX GPU setup and power monitoring
- ✅ **Plug-and-Play**: Easy to swap different algorithms
- ✅ **Comprehensive Metrics**: Quality (PSNR, SSIM) + Energy + Performance
- ✅ **Visualization**: Automatic side-by-side comparison images
- ✅ **Organized Storage**: Timestamped experiment directories
- ✅ **Fair Comparison**: Standardized test images ensure accurate evaluation
- ✅ **ISling Integration**: Multi-layer Ising denoiser with format conversion

## Example Workflow

```bash
# 1. Test integration
python test_integration.py

# 2. Prepare test images
python prepare_test_images.py

# 3. Run full comparison
python main_comparison.py

# 4. Analyze results
# Check data/output/*_summary.json for comparisons
# View comparison.png files for visual quality
```

## See Also

- `pipeline/USAGE.md` - Detailed usage guide
- `pipeline/README.md` - Pipeline architecture
- `LEARNINGS.md` - What we learned from implementation

