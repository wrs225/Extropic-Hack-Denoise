"""GPU power monitoring and benchmarking utilities."""

import time
import threading
from contextlib import contextmanager
from typing import Optional, Dict, List, Callable, Any
import numpy as np

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    try:
        # Try nvidia-ml-py (newer package name)
        import pynvml
        PYNVML_AVAILABLE = True
    except ImportError:
        PYNVML_AVAILABLE = False
        print("Warning: pynvml/nvidia-ml-py not available. Power monitoring will be disabled.")


class GPUPowerMonitor:
    """Context manager for monitoring GPU power consumption."""
    
    def __init__(self, gpu_id: int = 0, sample_rate: float = 0.1):
        """
        Initialize GPU power monitor.
        
        Args:
            gpu_id: GPU device ID
            sample_rate: Sampling rate in seconds (default 100ms = 0.1s)
        """
        self.gpu_id = gpu_id
        self.sample_rate = sample_rate
        self.power_samples: List[float] = []
        self.timestamps: List[float] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            except Exception as e:
                print(f"Warning: Failed to initialize NVML: {e}")
                self.handle = None
        else:
            self.handle = None
    
    def _monitor_loop(self):
        """Internal monitoring loop."""
        while self.is_monitoring:
            if self.handle is not None:
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert mW to W
                    timestamp = time.time()
                    self.power_samples.append(power)
                    self.timestamps.append(timestamp)
                except Exception:
                    # If power reading fails, use 0
                    timestamp = time.time()
                    self.power_samples.append(0.0)
                    self.timestamps.append(timestamp)
            else:
                # Fallback: use 0 if NVML not available
                timestamp = time.time()
                self.power_samples.append(0.0)
                self.timestamps.append(timestamp)
            
            time.sleep(self.sample_rate)
    
    def __enter__(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.is_monitoring = True
        self.power_samples = []
        self.timestamps = []
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring."""
        self.is_monitoring = False
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=1.0)
        self.end_time = time.time()
    
    @property
    def total_energy_j(self) -> float:
        """Calculate total energy consumption in Joules."""
        if len(self.power_samples) < 2:
            return 0.0
        
        # Integrate power over time
        power_array = np.array(self.power_samples)
        time_array = np.array(self.timestamps)
        
        # Calculate energy by trapezoidal integration
        if len(time_array) > 1:
            dt = np.diff(time_array)
            # Use average power over each interval
            avg_power = (power_array[:-1] + power_array[1:]) / 2.0
            energy = np.sum(avg_power * dt)
            return float(energy)
        
        return 0.0
    
    @property
    def avg_power_w(self) -> float:
        """Calculate average power consumption in Watts."""
        if len(self.power_samples) == 0:
            return 0.0
        return float(np.mean(self.power_samples))
    
    @property
    def max_power_w(self) -> float:
        """Get maximum power consumption in Watts."""
        if len(self.power_samples) == 0:
            return 0.0
        return float(np.max(self.power_samples))
    
    @property
    def min_power_w(self) -> float:
        """Get minimum power consumption in Watts."""
        if len(self.power_samples) == 0:
            return 0.0
        return float(np.min(self.power_samples))
    
    @property
    def duration_s(self) -> float:
        """Get monitoring duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def export_trace(self, filepath: str):
        """Export power trace to CSV file."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'power_w'])
            for ts, power in zip(self.timestamps, self.power_samples):
                writer.writerow([ts, power])


def benchmark_function(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    n_warmup: int = 5,
    n_runs: int = 20,
    gpu_id: int = 0,
) -> Dict[str, Any]:
    """
    Benchmark a function with GPU power monitoring.
    
    Args:
        func: Function to benchmark
        args: Positional arguments for function
        kwargs: Keyword arguments for function
        n_warmup: Number of warmup runs (excluded from measurements)
        n_runs: Number of benchmark runs
        gpu_id: GPU device ID
    
    Returns:
        Dictionary with benchmark results
    """
    if kwargs is None:
        kwargs = {}
    
    # Warmup runs
    print(f"Running {n_warmup} warmup iterations...")
    for _ in range(n_warmup):
        _ = func(*args, **kwargs)
    
    # Benchmark runs
    print(f"Running {n_runs} benchmark iterations...")
    times = []
    energies = []
    powers = []
    
    for i in range(n_runs):
        monitor = GPUPowerMonitor(gpu_id=gpu_id)
        
        with monitor:
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
        
        times.append((end - start) * 1000.0)  # Convert to ms
        energies.append(monitor.total_energy_j)
        powers.append(monitor.avg_power_w)
        
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{n_runs} runs")
    
    # Calculate statistics
    results = {
        'mean_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'mean_energy_j': float(np.mean(energies)),
        'std_energy_j': float(np.std(energies)),
        'mean_power_w': float(np.mean(powers)),
        'std_power_w': float(np.std(powers)),
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'min_energy_j': float(np.min(energies)),
        'max_energy_j': float(np.max(energies)),
        'n_runs': n_runs,
        'n_warmup': n_warmup,
    }
    
    return results

