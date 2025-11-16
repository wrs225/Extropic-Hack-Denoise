# Implementation Guide for Cursor Agent

This guide provides step-by-step instructions for implementing the JAX bilateral filter baseline using Cursor AI agent.

## Quick Start Commands

```bash
# 1. Navigate to project directory
cd ~/thrml-denoising-baseline

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify JAX GPU support
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# 5. Download test image
mkdir -p data/raw
wget https://homepages.cae.wisc.edu/~ece533/images/lena.png -O data/raw/test.png

# 6. Run baseline
python experiments/run_baseline.py \
    --image data/raw/test.png \
    --noise-level 25 \
    --save-images \
    --verbose
```

## Implementation Checklist for Cursor

Use this checklist to guide Cursor through the implementation:

### Phase 1: Environment Setup ✓
- [x] Project structure created
- [ ] Virtual environment activated
- [ ] Dependencies installed from requirements.txt
- [ ] JAX GPU support verified
- [ ] Test images downloaded

**Cursor prompt:**
```
Set up the Python environment for this JAX bilateral filter project:
1. Create and activate virtual environment
2. Install requirements.txt
3. Verify JAX can see the GPU
4. Download a test image to data/raw/
```

### Phase 2: Core Implementation ✓
- [x] Bilateral filter implementation (src/filters/bilateral.py)
- [x] Power monitoring utilities (src/benchmarks/power_metrics.py)
- [x] Image I/O utilities (src/utils/image_io.py)
- [x] Noise generation utilities (src/utils/noise_gen.py)

**Cursor prompt:**
```
Review the bilateral filter implementation in src/filters/bilateral.py.
Check if:
1. JAX imports are correct
2. JIT compilation is properly used
3. The filter logic is mathematically correct
4. Error handling is adequate
```

### Phase 3: Testing
- [ ] Test bilateral filter on sample image
- [ ] Verify power monitoring works
- [ ] Check noise generation produces expected results
- [ ] Validate PSNR/SSIM calculations

**Cursor prompt:**
```
Create and run basic tests for:
1. Load an image and apply bilateral filter
2. Monitor GPU power during filtering
3. Add Gaussian noise and calculate PSNR
4. Save results to verify output

Use the example in data/raw/ folder.
```

### Phase 4: Benchmarking
- [ ] Run full benchmark with experiments/run_baseline.py
- [ ] Verify results are saved to JSON
- [ ] Check denoised images look correct
- [ ] Validate energy measurements are reasonable

**Cursor prompt:**
```
Run the baseline benchmark script:
python experiments/run_baseline.py --image data/raw/test.png --noise-level 25 --save-images --verbose

Then verify:
1. JSON results are created in data/results/
2. Images are saved correctly
3. Energy values are in expected range (1-5J for 512x512)
4. PSNR values are reasonable (25-35 dB)
```

### Phase 5: Hackathon Integration
- [ ] Understand thermodynamic computing interface
- [ ] Implement comparison script
- [ ] Create visualization notebook
- [ ] Document energy efficiency findings

**Cursor prompt:**
```
Help me prepare for comparing this baseline against thermodynamic computing:
1. Show me how to call the bilateral filter with custom parameters
2. Explain the power monitoring context manager
3. Create a template for the thermodynamic comparison
4. Set up a Jupyter notebook for visualizing results
```

## Common Cursor Prompts

### Debugging

**"Fix JAX GPU issues"**
```
I'm getting an error about JAX not finding the GPU. Help me:
1. Check CUDA installation
2. Verify JAX CUDA version matches system CUDA
3. Test with a simple JAX GPU operation
4. Reinstall JAX with correct CUDA version if needed
```

**"Power monitoring returns zero"**
```
The GPU power monitoring is returning zero watts. Debug by:
1. Checking if nvidia-smi works
2. Verifying pynvml installation
3. Testing with a simple GPU workload
4. Checking permissions for power monitoring
```

### Optimization

**"Make bilateral filter faster"**
```
Optimize the bilateral filter implementation:
1. Review the vectorized version
2. Check if JIT compilation is actually happening
3. Profile to find bottlenecks
4. Suggest optimizations for the convolution operation
```

**"Reduce memory usage"**
```
The bilateral filter is using too much GPU memory for large images. Help:
1. Implement tiled/chunked processing
2. Reduce precision if acceptable
3. Clear JAX cache between runs
4. Add memory usage monitoring
```

### Extension

**"Add new noise type"**
```
Add support for [noise type] to the noise generator:
1. Implement the noise function following existing pattern
2. Add to NoiseGenerator class
3. Create convenience function
4. Update tests and documentation
```

**"Support different image formats"**
```
Extend image_io.py to support [format]:
1. Add format detection
2. Handle special cases (16-bit, etc.)
3. Update load_image and save_image
4. Add format-specific tests
```

## File Generation Prompts

### Create __init__.py files

**Cursor prompt:**
```
Generate appropriate __init__.py files for all Python packages in src/:
- src/__init__.py
- src/filters/__init__.py
- src/benchmarks/__init__.py
- src/utils/__init__.py

Each should export the main classes/functions from that module.
```

### Create test files

**Cursor prompt:**
```
Create pytest test files for:
1. tests/test_bilateral.py - Test bilateral filter
2. tests/test_power_metrics.py - Test power monitoring
3. tests/test_image_io.py - Test image loading/saving
4. tests/test_noise_gen.py - Test noise generation

Include fixtures for sample images and basic functionality tests.
```

## Verification Commands

**After each phase, run these to verify:**

```bash
# Phase 1: Environment
python -c "import jax; import numpy; import PIL; print('✓ All imports successful')"

# Phase 2: Core implementation
python -c "from src.filters.bilateral import BilateralFilterJAX; print('✓ Filter import successful')"

# Phase 3: Testing
python -c "
from src.filters.bilateral import bilateral_filter
from src.utils.image_io import load_image
img = load_image('data/raw/test.png')
result = bilateral_filter(img)
print(f'✓ Filter works: {result.shape}')
"

# Phase 4: Benchmarking
ls data/results/*.json && echo "✓ Benchmark results saved"

# Phase 5: Final check
python experiments/run_baseline.py --help && echo "✓ Experiment script ready"
```

## Expected Behavior

**Successful baseline run should produce:**

```
Loading images...
Loaded 1 images

Filter parameters: {'d': 9, 'sigma_color': 75.0, 'sigma_space': 75.0}
Noise levels: [25.0]
Benchmark: 5 warmup + 20 runs

============================================================
Processing image 1/1: test
============================================================

============================================================
Benchmarking: noise_level=25.0
============================================================
Running 5 warmup + 20 benchmark iterations...
Running 5 warmup iterations...
Running 20 measured iterations...
  Completed 5/20 runs
  Completed 10/20 runs
  Completed 15/20 runs
  Completed 20/20 runs

Results:
  Energy:     2.456 ± 0.123 J
  Power:      97.52 ± 3.21 W
  Time:       25.18 ± 1.05 ms
  PSNR:       29.84 dB
  SSIM:       0.8523
  Energy/px:  9.34 nJ
  Efficiency: 12.15 PSNR/J
```

## Troubleshooting Guide for Cursor

### Issue: ImportError for JAX
**Cursor fix prompt:**
```
I'm getting "ImportError: No module named 'jax'". Help me:
1. Check if virtual environment is activated
2. Reinstall jax with correct CUDA version
3. Verify installation with pip list | grep jax
```

### Issue: CUDA version mismatch
**Cursor fix prompt:**
```
JAX says CUDA version mismatch. Fix by:
1. Check system CUDA: nvcc --version
2. Install matching JAX version
3. Or install JAX without CUDA and use CPU for testing
```

### Issue: Bilateral filter is slow
**Cursor fix prompt:**
```
The bilateral filter is taking >1 second per image. Debug:
1. Verify JIT compilation is happening (check for compilation messages)
2. Ensure GPU is being used (check nvidia-smi during run)
3. Check if using vectorized implementation
4. Profile with JAX profiler
```

### Issue: Energy measurements seem wrong
**Cursor fix prompt:**
```
GPU energy readings are unusual (too high/low/zero). Diagnose:
1. Test nvidia-smi power monitoring manually
2. Check if other processes are using GPU
3. Verify sampling rate is appropriate
4. Compare with baseline power draw (idle GPU)
```

## Next Steps After Implementation

1. **Run comprehensive baseline**
   ```bash
   python experiments/run_baseline.py \
       --image-dir data/raw/ \
       --noise-levels 10 25 50 \
       --n-runs 20 \
       --save-results data/results/comprehensive_baseline.json \
       --save-images
   ```

2. **Analyze results**
   ```bash
   jupyter notebook notebooks/visualization.ipynb
   ```

3. **Implement thermodynamic approach**
   - Use power monitoring infrastructure
   - Compare against baseline metrics
   - Focus on energy efficiency

4. **Present findings**
   - Energy per image comparison
   - Quality metrics comparison
   - Energy efficiency analysis

## Resources

- **JAX documentation**: https://jax.readthedocs.io/
- **Bilateral filter theory**: https://en.wikipedia.org/wiki/Bilateral_filter
- **CUDA setup**: https://developer.nvidia.com/cuda-downloads
- **Extropic THRML**: [Your hackathon resources]

## Getting Help

**For Cursor agent:**
- Be specific about errors (include full stack trace)
- Reference specific files when asking for modifications
- Ask for explanations of complex parts
- Request step-by-step debugging when stuck

**Example good prompts:**
✓ "The bilateral_filter_kernel_vectorized function in src/filters/bilateral.py line 127 is raising a shape mismatch error. Debug and fix."
✓ "Explain how the power integration works in src/benchmarks/power_metrics.py lines 200-210."
✓ "Add docstring examples to all functions in src/utils/noise_gen.py"

**Example bad prompts:**
✗ "Fix the code"
✗ "It doesn't work"
✗ "Make it better"