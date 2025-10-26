#!/usr/bin/env python3
"""
Setup script for Brain Connectivity-Based Cognitive Prediction project.

This script:
1. Checks Python version
2. Creates necessary directory structure
3. Installs dependencies
4. Verifies installations
5. Downloads data (optional)
6. Provides next steps

Author: Based on Dhamala et al. (2021)
"""

import sys
import os
import subprocess
from pathlib import Path
import platform


def print_header(text: str) -> None:
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print_header("Step 1: Checking Python Version")
    
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    print(f"Required Python version: {required_version[0]}.{required_version[1]}+")
    print(f"Current Python version: {current_version[0]}.{current_version[1]}")
    
    if current_version >= required_version:
        print("✓ Python version is compatible")
        return True
    else:
        print(f"✗ Python version is too old. Please upgrade to Python {required_version[0]}.{required_version[1]} or higher.")
        return False


def create_directory_structure() -> None:
    """Create project directory structure."""
    print_header("Step 2: Creating Directory Structure")
    
    directories = [
        "data/raw",
        "data/processed",
        "outputs/models",
        "outputs/results",
        "outputs/figures",
        "outputs/reports",
        "outputs/logs",
        "notebooks",
        "tests"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\nDirectory structure created successfully!")


def install_dependencies() -> bool:
    """Install required Python packages."""
    print_header("Step 3: Installing Dependencies")
    
    print("Installing packages from requirements.txt...")
    print("This may take a few minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
        print("✓ pip upgraded successfully\n")
        
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✓ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing dependencies: {e}")
        print("Please install manually using: pip install -r requirements.txt")
        return False


def verify_installations() -> bool:
    """Verify that all required packages are installed."""
    print_header("Step 4: Verifying Installations")
    
    required_packages = [
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "nibabel",
        "matplotlib",
        "seaborn",
        "yaml",
        "tqdm",
        "joblib"
    ]
    
    all_installed = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            all_installed = False
    
    if all_installed:
        print("\n✓ All packages verified successfully!")
    else:
        print("\n✗ Some packages are missing. Please check installation.")
    
    return all_installed


def check_system_info() -> None:
    """Display system information."""
    print_header("System Information")
    
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Platform: {platform.machine()}")
    
    try:
        import numpy as np
        print(f"NumPy Version: {np.__version__}")
        
        import sklearn
        print(f"Scikit-learn Version: {sklearn.__version__}")
        
        import pandas as pd
        print(f"Pandas Version: {pd.__version__}")
    except ImportError:
        pass


def display_next_steps() -> None:
    """Display instructions for next steps."""
    print_header("Setup Complete! Next Steps:")
    
    print("""
1. Review Configuration:
   - Edit config.yaml to adjust parameters
   - Set use_synthetic: true (default) or false (if you have HCP data)

2. Run the Pipeline:
   python run_pipeline.py --config config.yaml

3. Quick Test (single iteration):
   python run_pipeline.py --config config.yaml --quick

4. View Results:
   - Results: outputs/results/
   - Figures: outputs/figures/
   - Report: outputs/reports/analysis_report.html

5. Explore with Jupyter:
   jupyter notebook notebooks/

For help:
   python run_pipeline.py --help

For more information, see README.md
    """)


def main():
    """Main setup function."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  Brain Connectivity-Based Cognitive Prediction".center(78) + "║")
    print("║" + "  Setup Script".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Create directories
    create_directory_structure()
    
    # Step 3: Install dependencies
    install_success = install_dependencies()
    
    # Step 4: Verify installations
    if install_success:
        verify_success = verify_installations()
    else:
        verify_success = False
    
    # Display system info
    check_system_info()
    
    # Display next steps
    if verify_success:
        display_next_steps()
        print("\n✓ Setup completed successfully!\n")
    else:
        print("\n✗ Setup completed with errors. Please review the output above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
