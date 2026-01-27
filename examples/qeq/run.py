# SPDX-License-Identifier: LGPL-3.0-or-later
"""
QEq Examples Main Entry Point

This script serves as the main entry point for QEq examples.
It runs the comprehensive example suite that demonstrates all QEq features.
"""

import sys
from pathlib import Path

# Import and run the comprehensive example suite
try:
    from run_all import main as run_all_main

    if __name__ == "__main__":
        print("QEq Examples - torch-admp")
        print("=" * 50)
        print("This script will run all QEq examples to demonstrate")
        print("the full capabilities of the torch-admp QEq implementation.")
        print("For individual examples, see the files in this directory.")
        print("=" * 50)

        # Run all examples
        success = run_all_main()

        if success:
            print("\n🎉 All QEq examples completed successfully!")
            print("\nTo run individual examples, use:")
            print("  python basic_qeq.py")
            print("  python matrix_inversion.py")
            print("  python optimization_methods.py")
            print("  python advanced_parameters.py")
            print("  python convergence_criteria.py")
            print("  python batch_processing.py")
            print("  python jit_compilation.py")
            print("  python hessian_calculation.py")
            print("\nFor more information, see README.md")
        else:
            print("\n⚠ Some examples failed. Please check the error messages above.")
            sys.exit(1)

except ImportError as e:
    print(f"Error importing run_all.py: {e}")
    print("Please ensure you're running this script from the examples/qeq directory")
    sys.exit(1)
