# SPDX-License-Identifier: LGPL-3.0-or-later
"""
PME Examples Runner for torch-admp

This script runs all PME examples to demonstrate the full functionality
of the PME implementation in torch-admp.
"""

import sys
from pathlib import Path

# Import the run_all module
from run_all import main as run_all_main


def main():
    """Main entry point for PME examples."""
    # Check if we're in the right directory
    current_dir = Path(__file__).parent
    if not (current_dir / "run_all.py").exists():
        print("Error: run_all.py not found in the same directory.")
        print("Please make sure all example files are present.")
        return 1

    print("PME Examples for torch-admp")
    print("=" * 60)
    print("This script demonstrates the comprehensive PME functionality in torch-admp.")
    print("Each example focuses on a specific aspect of the PME implementation.")
    print("\nAvailable individual examples:")
    print("  - basic_pme.py: Basic PME usage")
    print("  - advanced_parameters.py: Advanced PME parameters")
    print("  - energy_components.py: Energy component access")
    print("  - jit_compilation.py: JIT compilation")
    print("  - batch_processing.py: Batch processing")
    print("  - error_handling.py: Error handling")
    print("  - slab_vs_3d.py: 3D PBC vs 2D slab correction")
    print("  - setup_ewald.py: Setup Ewald parameters")
    print("\nRunning all examples...")

    # Run all examples
    return run_all_main()


if __name__ == "__main__":
    sys.exit(main())
