"""Terminal-based visualization using Rich library."""

from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

console = Console()


def display_metrics(
    metrics: Dict[str, Any],
    title: str = "Benchmark Results",
) -> None:
    """
    Display metrics in a formatted table.
    
    Args:
        metrics: Dictionary with metric values
        title: Title for the display
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_column("Unit", style="yellow")
    
    # Add metrics
    if 'mean_time_ms' in metrics:
        table.add_row("Processing Time", f"{metrics['mean_time_ms']:.2f} ± {metrics.get('std_time_ms', 0):.2f}", "ms")
    
    if 'mean_energy_j' in metrics:
        table.add_row("Energy", f"{metrics['mean_energy_j']:.3f} ± {metrics.get('std_energy_j', 0):.3f}", "J")
    
    if 'mean_power_w' in metrics:
        table.add_row("Power", f"{metrics['mean_power_w']:.2f} ± {metrics.get('std_power_w', 0):.2f}", "W")
    
    if 'psnr' in metrics:
        table.add_row("PSNR", f"{metrics['psnr']:.2f}", "dB")
    
    if 'ssim' in metrics:
        table.add_row("SSIM", f"{metrics['ssim']:.4f}", "")
    
    if 'psnr_per_joule' in metrics:
        table.add_row("PSNR/Joule", f"{metrics['psnr_per_joule']:.2f}", "dB/J")
    
    if 'energy_per_pixel_nj' in metrics:
        table.add_row("Energy/Pixel", f"{metrics['energy_per_pixel_nj']:.2f}", "nJ")
    
    console.print(table)


def visualize_power_consumption(
    timestamps: List[float],
    power_samples: List[float],
    title: str = "Power Consumption Over Time",
) -> None:
    """
    Visualize power consumption as a simple text plot.
    
    Args:
        timestamps: List of timestamps
        power_samples: List of power values in Watts
        title: Title for the plot
    """
    if len(power_samples) == 0:
        console.print("[yellow]No power data to visualize[/yellow]")
        return
    
    # Create simple ASCII plot
    max_power = max(power_samples)
    min_power = min(power_samples)
    avg_power = sum(power_samples) / len(power_samples)
    
    # Create a simple bar chart representation
    console.print(f"\n[bold]{title}[/bold]")
    console.print(f"Min: {min_power:.2f} W | Avg: {avg_power:.2f} W | Max: {max_power:.2f} W\n")
    
    # Simple text-based visualization
    # Sample every 10th point for display
    step = max(1, len(power_samples) // 50)
    sampled_power = power_samples[::step]
    
    # Create bars
    bar_chart = ""
    for power in sampled_power:
        # Normalize to 0-50 characters
        bar_length = int((power - min_power) / (max_power - min_power + 1e-8) * 50)
        bar = "█" * bar_length
        bar_chart += f"{power:6.1f} W |{bar}\n"
    
    console.print(bar_chart)


def display_comparison(
    baseline_metrics: Dict[str, Any],
    isling_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Display side-by-side comparison of baseline vs ISling.
    
    Args:
        baseline_metrics: Metrics from baseline implementation
        isling_metrics: Optional metrics from ISling implementation
    """
    table = Table(title="Baseline vs ISling Comparison", show_header=True, header_style="bold magenta")
    
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Baseline", style="green")
    if isling_metrics:
        table.add_column("ISling", style="blue")
        table.add_column("Difference", style="yellow")
    
    # Add comparison rows
    if 'mean_energy_j' in baseline_metrics:
        baseline_energy = baseline_metrics['mean_energy_j']
        if isling_metrics and 'mean_energy_j' in isling_metrics:
            isling_energy = isling_metrics['mean_energy_j']
            diff = ((baseline_energy - isling_energy) / baseline_energy) * 100
            table.add_row(
                "Energy (J)",
                f"{baseline_energy:.3f}",
                f"{isling_energy:.3f}",
                f"{diff:+.1f}%"
            )
        else:
            table.add_row("Energy (J)", f"{baseline_energy:.3f}", "-", "-")
    
    if 'psnr' in baseline_metrics:
        baseline_psnr = baseline_metrics['psnr']
        if isling_metrics and 'psnr' in isling_metrics:
            isling_psnr = isling_metrics['psnr']
            diff = isling_psnr - baseline_psnr
            table.add_row(
                "PSNR (dB)",
                f"{baseline_psnr:.2f}",
                f"{isling_psnr:.2f}",
                f"{diff:+.2f} dB"
            )
        else:
            table.add_row("PSNR (dB)", f"{baseline_psnr:.2f}", "-", "-")
    
    if 'mean_time_ms' in baseline_metrics:
        baseline_time = baseline_metrics['mean_time_ms']
        if isling_metrics and 'mean_time_ms' in isling_metrics:
            isling_time = isling_metrics['mean_time_ms']
            diff = ((baseline_time - isling_time) / baseline_time) * 100
            table.add_row(
                "Time (ms)",
                f"{baseline_time:.2f}",
                f"{isling_time:.2f}",
                f"{diff:+.1f}%"
            )
        else:
            table.add_row("Time (ms)", f"{baseline_time:.2f}", "-", "-")
    
    console.print(table)


def show_progress_bar(total: int, description: str = "Processing"):
    """
    Create and return a progress bar context manager.
    
    Args:
        total: Total number of items to process
        description: Description text
    
    Returns:
        Progress context manager
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )
    
    task = progress.add_task(description, total=total)
    
    return progress, task

