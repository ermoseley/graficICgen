#!/usr/bin/env python3
"""
Demonstration script for comparing different turbulence spectrum types.

This script generates turbulent ICs with different spectrum types and plots
their power spectra for comparison. It demonstrates the new power law option
in turb.py and the spectrum plotting functionality.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    print(f"Success: {result.stdout.strip()}")
    return result

def main():
    print("=== Turbulence Spectrum Comparison Demo ===\n")
    
    # Create output directory
    output_dir = Path("demo_spectra")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir.absolute()}\n")
    
    # Generate parabolic spectrum ICs
    print("1. Generating parabolic spectrum ICs...")
    cmd_parabolic = [
        sys.executable, "turb.py", "5",  # level 5 (32x32 grid)
        "--ndim", "2",
        "--kmin", "2",
        "--kmax", "8",
        "--spectrum", "parabolic",
        "--vrms", "0.1",
        "--outdir", str(output_dir / "parabolic")
    ]
    run_command(cmd_parabolic, "Parabolic spectrum generation")
    
    # Generate power law spectrum ICs (Kolmogorov)
    print("\n2. Generating power law spectrum ICs (Kolmogorov slope)...")
    cmd_powerlaw_kolmogorov = [
        sys.executable, "turb.py", "5",  # level 5 (32x32 grid)
        "--ndim", "2",
        "--kmin", "2",
        "--kmax", "8",
        "--spectrum", "power_law",
        "--slope", "-1.67",
        "--vrms", "0.1",
        "--outdir", str(output_dir / "powerlaw_kolmogorov")
    ]
    run_command(cmd_powerlaw_kolmogorov, "Power law (Kolmogorov) generation")
    
    # Generate power law spectrum ICs (steep slope)
    print("\n3. Generating power law spectrum ICs (steep slope)...")
    cmd_powerlaw_steep = [
        sys.executable, "turb.py", "5",  # level 5 (32x32 grid)
        "--ndim", "2",
        "--kmin", "2",
        "--kmax", "8",
        "--spectrum", "power_law",
        "--slope", "-2.0",
        "--vrms", "0.1",
        "--outdir", str(output_dir / "powerlaw_steep")
    ]
    run_command(cmd_powerlaw_steep, "Power law (steep) generation")
    
    # Plot individual spectra
    print("\n4. Plotting individual spectra...")
    
    # Parabolic
    cmd_plot_parabolic = [
        sys.executable, "plot_spectrum.py", str(output_dir / "parabolic"),
        "--save", str(output_dir / "parabolic_spectrum.png"),
        "--no-show"
    ]
    run_command(cmd_plot_parabolic, "Parabolic spectrum plotting")
    
    # Power law Kolmogorov
    cmd_plot_kolmogorov = [
        sys.executable, "plot_spectrum.py", str(output_dir / "powerlaw_kolmogorov"),
        "--save", str(output_dir / "kolmogorov_spectrum.png"),
        "--no-show"
    ]
    run_command(cmd_plot_kolmogorov, "Kolmogorov spectrum plotting")
    
    # Power law steep
    cmd_plot_steep = [
        sys.executable, "plot_spectrum.py", str(output_dir / "powerlaw_steep"),
        "--save", str(output_dir / "steep_spectrum.png"),
        "--no-show"
    ]
    run_command(cmd_plot_steep, "Steep spectrum plotting")
    
    # Plot comparison
    print("\n5. Creating comparison plot...")
    cmd_plot_comparison = [
        sys.executable, "plot_spectrum.py",
        str(output_dir / "parabolic"),
        str(output_dir / "powerlaw_kolmogorov"),
        str(output_dir / "powerlaw_steep"),
        "--labels", "Parabolic", "Power Law (-5/3)", "Power Law (-2.0)",
        "--save", str(output_dir / "spectrum_comparison.png"),
        "--no-show"
    ]
    run_command(cmd_plot_comparison, "Spectrum comparison plotting")
    
    print(f"\n=== Demo Complete! ===")
    print(f"Generated ICs and plots in: {output_dir.absolute()}")
    print(f"\nFiles created:")
    print(f"  - Parabolic spectrum ICs: {output_dir / 'parabolic'}")
    print(f"  - Power law (Kolmogorov) ICs: {output_dir / 'powerlaw_kolmogorov'}")
    print(f"  - Power law (steep) ICs: {output_dir / 'powerlaw_steep'}")
    print(f"  - Individual spectrum plots: *.png")
    print(f"  - Comparison plot: spectrum_comparison.png")
    print(f"\nYou can now:")
    print(f"  1. View the plots to see the spectral differences")
    print(f"  2. Use these ICs in mini-ramses simulations")
    print(f"  3. Compare the behavior of different spectrum types")

if __name__ == "__main__":
    main()
