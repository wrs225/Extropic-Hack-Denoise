# Quick Start Guide

Get the JAX bilateral filter baseline running in 5 minutes.

## Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.8+
- CUDA 11.0+ installed
- 8GB+ GPU memory recommended

## Installation

```bash
# 1. Navigate to project directory
cd /home/claude/thrml-denoising-baseline

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate  # On Windows

# 3. Install package in development mode
pip install -e .

# OR just install requirements
pip install -r requirements.txt

# 4. Verify JAX GPU support
python -c "import jax; print('Devices:', jax.devices())"
# Should show: Devices: [cuda(id=0)] or similar
```

## Quick Test

```bash
# Download a test image
mkdir -p data/raw
wget https://homepages.cae.wisc.edu/~ece533/images/lena.png -O data/raw/lena.png

# Run baseline (takes ~1 minute)
python experiments/run_baseline.py \
    --image data/raw/lena.png \
    --noise-level 25 \
    --n-runs 5 \
    --save-images \
    --verbose

# Check results
ls data/results/
```

## Expected Output

```
Loading images...
Loaded 1 images

Filter parameters: {'d': 9, 'sigma_color': 75.0, 'sigma_space': 75.0}
Noise levels: [25.0]
Benchmark: 5 warmup + 5 runs

============================================================
Processing image 1/1: lena
============================================================

Running 5 warmup + 5 benchmark iterations...
Running 5 warmup iterations...
Running 5 measured iterations...
  Completed 5/5 runs

Results:
  Energy:     2.456 ± 0.123 J
  Power:      97.52 ± 3.21 W
  Time:       25.18 ± 1.05 ms
  PSNR:       29.84 dB
  SSIM:       0.8523
  Energy/px:  9.34 nJ
  Efficiency: 12.15 PSNR/J

Results saved to: data/results/baseline_20250116_143022.json
```

## Usage Examples

### Basic Denoising

```python
from src.filters.bilateral import bilateral_filter
from src.utils.image_io import load_image, save_image
from src.utils.noise_gen import add_gaussian_noise

# Load image
clean = load_image('data/raw/test.png', normalize=True)

# Add noise
noisy = add_gaussian_noise(clean, sigma=25, seed=42)

# Denoise
denoised = bilateral_filter(noisy, d=9, sigma_color=75, sigma_space=75)

# Save
save_image(denoised, 'denoised.png')
```

### With Power Monitoring

```python
from src.filters.bilateral import BilateralFilterJAX
from src.benchmarks.power_metrics import GPUPowerMonitor

# Create filter
bf = BilateralFilterJAX(d=9, sigma_color=75, sigma_space=75)

# Monitor power
monitor = GPUPowerMonitor(gpu_id=0)

with monitor:
    result = bf.denoise(noisy)

print(f"Energy: {monitor.total_energy_j:.2f}J")
print(f"Power: {monitor.avg_power_w:.2f}W")
```

### Batch Processing

```python
from src.utils.image_io import load_images_from_dir

# Load multiple images
images = load_images_from_dir('data/raw/', normalize=True)

# Process batch
bf = BilateralFilterJAX(d=9, sigma_color=75, sigma_space=75)
results = bf.denoise_batch(images)
```

## Common Issues

### Issue: "No CUDA devices found"

**Solution:**
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall JAX with CUDA support
pip uninstall jax jaxlib
pip install jax[cuda12]==0.4.23  # For CUDA 12
# OR
pip install jax[cuda11]==0.4.23  # For CUDA 11
```

### Issue: "Out of memory"

**Solution:**
```python
# Use smaller images
from src.utils.image_io import resize_image
img_small = resize_image(img, (256, 256))

# OR reduce filter diameter
bf = BilateralFilterJAX(d=5)  # Instead of d=9
```

### Issue: "nvidia-smi permission denied"

**Solution:**
```bash
# Add user to video group
sudo usermod -a -G video $USER
# Log out and back in
```

## Next Steps

1. **Run comprehensive benchmark:**
   ```bash
   python experiments/run_baseline.py \
       --image-dir data/raw/ \
       --noise-levels 10 25 50 \
       --n-runs 20 \
       --save-results data/results/full_baseline.json \
       --save-images
   ```

2. **Explore in Jupyter:**
   ```bash
   jupyter notebook notebooks/
   ```

3. **Implement your thermodynamic approach:**
   - Use the power monitoring infrastructure
   - Compare against baseline results
   - Focus on energy efficiency

4. **Read the guides:**
   - `CURSOR_GUIDE.md` - Detailed Cursor agent instructions
   - `experiments/README.md` - Experiment workflow
   - `src/filters/README.md` - Filter implementation details

## Performance Targets

**Typical baseline on 512x512 RGB (RTX 3090):**
- Time: 15-30 ms
- Power: 80-120 W  
- Energy: 1.2-3.6 J
- PSNR (σ=25): 28-32 dB

**Your thermodynamic goal:**
- Energy < 1 J (60-80% savings)
- PSNR ≥ 28 dB (maintain quality)
- Demonstrate energy efficiency advantage

## Hackathon Workflow

```bash
# Day 1: Establish baseline
python experiments/run_baseline.py --image-dir data/raw/ --save-results baseline.json

# Day 2: Implement thermodynamic approach
# (Use your THRML code)

# Day 3: Compare and visualize
python experiments/compare_thermodynamic.py
jupyter notebook notebooks/visualization.ipynb

# Day 4: Optimize and polish
# Fine-tune parameters, create final plots
```

## Getting Help

- **Documentation**: Check README.md files in each directory
- **Cursor**: Use CURSOR_GUIDE.md for AI assistance
- **Issues**: Common problems and solutions in each module's README
- **Hackathon**: Consult THRML documentation and mentors

## Files Created

After running the quick test, you should see:
```
thrml-denoising-baseline/
├── data/
│   ├── raw/lena.png
│   └── results/
│       ├── baseline_*.json
│       └── lena_sigma25/
│           ├── clean.png
│           ├── noisy.png
│           ├── denoised.png
│           └── comparison.png
└── ...
```

## Ready to Go!

You now have a working JAX bilateral filter baseline with GPU power monitoring. Use this as your reference when developing and comparing your thermodynamic computing approach.

**Key metrics to track:**
- Energy per image (Joules)
- Image quality (PSNR, SSIM)
- Energy efficiency (PSNR/Joule)
- Processing time (milliseconds)

Good luck at the hackathon! 🚀