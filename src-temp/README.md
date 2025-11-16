# Extropic Baseline - Image Denoising Pipeline

Image denoising baseline implementation using JAX-based bilateral filter with GPU power monitoring.

## Setup

### Install Dependencies

From the `src-temp` directory:

```bash
cd src-temp
uv sync
```

This will install all dependencies including JAX. If JAX is already installed in the parent project's environment, `uv` should detect and reuse it.

### Verify GPU Access

```bash
uv run python -c "import jax; print(jax.devices())"
```

Should show GPU devices if available.

## Usage

### Run Toy Example

```bash
uv run python main.py --toy
```

This will:
- Create a synthetic 3-bit grayscale image
- Add Gaussian noise
- Apply bilateral filter
- Verify GPU usage and JIT compilation
- Test power monitoring
- Save example images to `s0_data/results/`

### Benchmark an Image

```bash
uv run python main.py --image <path-to-image> --noise-level 0.15 --n-runs 20 --save-images
```

Options:
- `--image`: Path to input image
- `--noise-level`: Noise standard deviation as fraction (default 0.15 = 15%)
- `--n-warmup`: Number of warmup runs (default 5)
- `--n-runs`: Number of benchmark runs (default 20)
- `--save-images`: Save output images

## Project Structure

```
src-temp/
├── pyproject.toml          # Dependencies (isolated from root)
├── main.py                 # Main pipeline entry point
├── s0_data/                # Data directory
│   ├── raw/                # Input images
│   └── results/            # Benchmark results and output images
├── s1_input/               # Data loading and noise generation
│   ├── loader.py           # Image loading, 3-bit quantization
│   └── noise_gen.py        # Reproducible noise generation
├── s2a_baseline/           # Baseline denoising algorithm
│   └── bilateral_filter.py # JAX bilateral filter (GPU-accelerated)
└── s3_results/             # Benchmarking and visualization
    ├── benchmark.py        # GPU power monitoring
    ├── metrics.py          # PSNR, SSIM, efficiency metrics
    └── visualizer.py       # Terminal-based visualization
```

## Dependencies

All dependencies are isolated in `src-temp/pyproject.toml`:
- `jax[cuda12]>=0.6.2` - JAX with CUDA support (shared with parent)
- `pynvml` - GPU power monitoring
- `rich` - Terminal visualization
- `numpy` - Array operations
- `pillow` - Image I/O

## Notes

- JAX is specified with the same version as the parent project to ensure compatibility
- Power monitoring requires NVIDIA GPU with NVML support
- Results are exported to JSON format for easy comparison with ISling results

