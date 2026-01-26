# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Run All PME Examples for torch-admp

This script runs all PME examples in sequence to demonstrate the full
functionality of the PME implementation in torch-admp.
"""

import sys
import subprocess
import time
from pathlib import Path


def run_example(example_name, example_path):
    """Run a single example and capture its output."""
    print(f"\n{'='*80}")
    print(f"RUNNING EXAMPLE: {example_name.upper()}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run the example and capture output
        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            timeout=60  # Timeout after 60 seconds
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ {example_name} completed successfully in {elapsed_time:.2f} seconds")
            print("\nOutput:")
            print(result.stdout)
        else:
            print(f"\n✗ {example_name} failed with return code {result.returncode}")
            print(f"\nError output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n✗ {example_name} timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"\n✗ {example_name} failed with exception: {e}")
        return False
    
    return True


def main():
    """Run all PME examples."""
    print("PME EXAMPLES FOR TORCH-ADMP")
    print("="*80)
    print("This script will run all PME examples to demonstrate the full functionality")
    print("of the PME implementation in torch-admp.")
    print("="*80)
    
    # List of examples to run
    examples = [
        ("Basic PME", "basic_pme.py"),
        ("Advanced Parameters", "advanced_parameters.py"),
        ("Energy Components", "energy_components.py"),
        ("JIT Compilation", "jit_compilation.py"),
        ("Batch Processing", "batch_processing.py"),
        ("Error Handling", "error_handling.py"),
        ("3D PBC vs 2D Slab", "slab_vs_3d.py"),
        ("Setup Ewald Parameters", "setup_ewald.py"),
    ]
    
    # Check if all example files exist
    current_dir = Path(__file__).parent
    missing_files = []
    
    for _, example_file in examples:
        example_path = current_dir / example_file
        if not example_path.exists():
            missing_files.append(str(example_path))
    
    if missing_files:
        print("Error: The following example files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease make sure all example files are present before running this script.")
        return 1
    
    # Run all examples
    successful_examples = []
    failed_examples = []
    
    total_start_time = time.time()
    
    for example_name, example_file in examples:
        example_path = current_dir / example_file
        
        if run_example(example_name, example_path):
            successful_examples.append(example_name)
        else:
            failed_examples.append(example_name)
    
    total_elapsed_time = time.time() - total_start_time
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total execution time: {total_elapsed_time:.2f} seconds")
    print(f"Successful examples: {len(successful_examples)}/{len(examples)}")
    
    if successful_examples:
        print("\n✓ Successful examples:")
        for example in successful_examples:
            print(f"  - {example}")
    
    if failed_examples:
        print("\n✗ Failed examples:")
        for example in failed_examples:
            print(f"  - {example}")
    
    # Return appropriate exit code
    if failed_examples:
        print(f"\n{len(failed_examples)} example(s) failed.")
        return 1
    else:
        print(f"\nAll {len(examples)} examples completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())