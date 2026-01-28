# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Run all QEq examples in sequence.

This script executes all QEq examples in the correct order,
providing a comprehensive demonstration of all features.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def run_example(script_name, description):
    """
    Run a single example script and capture its output.

    Parameters
    ----------
    script_name : str
        Name of the script to run
    description : str
        Description of what the script demonstrates

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print(f"Description: {description}")
    print(f"{'='*80}")

    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed_time = time.time() - start_time

        print(f"\n✓ {script_name} completed successfully in {elapsed_time:.2f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ {script_name} failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"\n✗ {script_name} failed with exception: {str(e)}")
        return False


def main():
    """
    Main function to run all QEq examples.
    """
    print("QEq Examples Suite")
    print("=" * 80)
    print("This script will run all QEq examples in sequence to demonstrate")
    print("the full capabilities of the torch-admp QEq implementation.")
    print("=" * 80)

    # Check if required data files exist
    required_files = ["qeq.pdb", "qeq.xml"]
    missing_files = []

    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"\n✗ Required data files not found: {', '.join(missing_files)}")
        print(
            "Please ensure you're running this script from the examples/qeq directory"
        )
        print("and that the required data files are present.")
        return False

    # List of examples to run in order
    examples = [
        ("basic_qeq.py", "Fundamental QEq usage and basic setup"),
        ("matrix_inversion.py", "Matrix inversion method for direct QEq solution"),
        (
            "optimization_methods.py",
            "Different optimization methods (LBFGS, quadratic)",
        ),
        ("advanced_parameters.py", "Advanced QEq parameters and configuration"),
        ("convergence_criteria.py", "Convergence criteria and monitoring"),
        ("batch_processing.py", "Batch processing of multiple configurations"),
        ("jit_compilation.py", "JIT compilation for performance optimization"),
        ("hessian_calculation.py", "Hessian matrix calculation and analysis"),
        # ("constraint_handling.py", "Constraint handling and vector projection"),
    ]

    # Track success/failure
    successful_examples = []
    failed_examples = []
    total_start_time = time.time()

    # Run each example
    for script_name, description in examples:
        if run_example(script_name, description):
            successful_examples.append(script_name)
        else:
            failed_examples.append(script_name)

            # Ask user whether to continue
            if failed_examples:
                print(f"\nContinue with remaining examples? (y/n): ", end="")
                try:
                    response = input().strip().lower()
                    if response not in ["y", "yes", ""]:
                        print("Stopping execution.")
                        break
                except (EOFError, KeyboardInterrupt):
                    print("\nStopping execution.")
                    break

    total_elapsed_time = time.time() - total_start_time

    # Print summary
    print(f"\n{'='*80}")
    print("QEq Examples Summary")
    print(f"{'='*80}")
    print(f"Total execution time: {total_elapsed_time:.2f} seconds")
    print(f"Successful examples: {len(successful_examples)}")
    print(f"Failed examples: {len(failed_examples)}")

    if successful_examples:
        print(f"\n✓ Successfully completed:")
        for example in successful_examples:
            print(f"  - {example}")

    if failed_examples:
        print(f"\n✗ Failed to complete:")
        for example in failed_examples:
            print(f"  - {example}")

    # Overall success status
    if not failed_examples:
        print(f"\n🎉 All QEq examples completed successfully!")
        print("\nYou have now seen a comprehensive demonstration of:")
        print("  • Basic QEq setup and usage")
        print("  • Matrix inversion and optimization methods")
        print("  • Advanced parameter configuration")
        print("  • Convergence criteria and monitoring")
        print("  • Batch processing capabilities")
        print("  • JIT compilation for performance")
        print("  • Hessian matrix analysis")
        print("  • Constraint handling")
        print("\nFor more details, see the README.md file in this directory.")
        return True
    else:
        print(f"\n⚠ Some examples failed. Please check the error messages above.")
        print(
            "You may need to install missing dependencies or fix configuration issues."
        )
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
