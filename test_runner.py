#!/usr/bin/env python3
"""
Test Runner for Camera Calibration Toolkit

Simple script to validate example scripts.
"""
import sys
import subprocess
import argparse
import os
import glob
from pathlib import Path

def validate_examples():
    """Validate example scripts by running them and catching errors."""
    
    # Automatically discover all Python example files
    example_patterns = [
        "examples/*.py",
        "examples/**/*.py"  # Include subdirectories if any
    ]
    
    examples = set()  # Use set to avoid duplicates
    for pattern in example_patterns:
        examples.update(glob.glob(pattern, recursive=True))
    
    # Remove any __pycache__ or other non-example files
    examples = [ex for ex in examples if not any(exclude in ex for exclude in ['__pycache__', '.pyc', '__init__.py'])]
    
    # Sort for consistent output
    examples.sort()
    
    print("============================================================")
    print("Example Scripts Validation")
    print("============================================================")
    print(f"Discovered {len(examples)} example files:")
    for example in examples:
        print(f"  - {example}")
    print("")
    
    success_count = 0
    total_count = len(examples)
    
    for example in examples:
        example_path = Path(example)
        print(f"\nTesting {example_path.name}...")
        
        if not example_path.exists():
            print(f"FAIL - Example not found: {example}")
            continue
            
        try:
            # Run the example with a timeout and proper encoding handling
            result = subprocess.run(
                [sys.executable, str(example_path)], 
                capture_output=True, 
                text=True, 
                timeout=300,  # 5 minute timeout
                encoding='utf-8',
                errors='replace',  # Replace problematic unicode characters
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8:replace'}
            )
            
            if result.returncode == 0:
                print(f"PASS - {example_path.name} - SUCCESS")
                success_count += 1
            else:
                print(f"FAIL - {example_path.name} - FAILED")
                print(f"   Return code: {result.returncode}")
                if result.stderr:
                    # Show first few lines of error
                    error_lines = result.stderr.strip().split('\n')[:3]
                    for line in error_lines:
                        print(f"   Error: {line}")
                        
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT - {example_path.name} - TIMEOUT (5 minutes)")
            
        except Exception as e:
            print(f"ERROR - {example_path.name} - EXCEPTION: {e}")
    
    print(f"\nExamples Summary: {success_count}/{total_count} passed")
    return success_count == total_count

def check_environment():
    """Check basic environment setup."""
    print("\n" + "="*60)
    print("Environment Check")
    print("="*60)
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    
    # Read requirements.txt to get all required packages
    requirements_file = Path("requirements.txt")
    essential_packages = []  # Will build from requirements
    
    if requirements_file.exists():
        print(f"Reading requirements from: {requirements_file}")
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before >= or == or other version specs)
                    package_name = line.split('>=')[0].split('==')[0].split('<')[0].strip()
                    if package_name:
                        essential_packages.append(package_name)
    else:
        print("WARNING: requirements.txt not found, checking essential packages only")
        # Add essential packages for examples
        essential_packages.extend(["numpy", "opencv-contrib-python", "PyYAML", "scipy", "Pillow", "flask"])
    
    print(f"Checking {len(essential_packages)} packages...")
    failed_packages = []
    
    # Check required packages
    for package in essential_packages:
        try:
            # Handle special package name mappings
            import_name = package.replace("-", "_")
            if package == "opencv-contrib-python":
                import_name = "cv2"
            elif package == "PyYAML":
                import_name = "yaml"
            elif package == "Pillow":
                import_name = "PIL"
            elif package == "Flask":
                import_name = "flask"
            elif package == "Werkzeug":
                import_name = "werkzeug"
            
            __import__(import_name)
            print(f"PASS: {package} - Available")
        except ImportError:
            print(f"MISSING: {package} - Missing")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\nTo install missing packages:")
        print(f"pip install {' '.join(failed_packages)}")
    
    # Check directory structure
    important_paths = ["examples/", "core/", "data/", "web/"]
    for path in important_paths:
        if Path(path).exists():
            print(f"PASS: {path} - Available")
        else:
            print(f"MISSING: {path} - Missing")
    
    # Check if we have sample data
    sample_data_path = Path("sample_data/")
    if sample_data_path.exists():
        print(f"PASS: sample_data/ - Available")
        # Count subdirectories in sample_data
        subdirs = [p for p in sample_data_path.iterdir() if p.is_dir()]
        print(f"   Found {len(subdirs)} sample data sets")
    else:
        print(f"WARNING: sample_data/ - Missing (examples may fail)")
    
    # Check core modules can be imported
    core_modules = ["core.calibration_patterns", "core.intrinsic_calibration", "core.eye_in_hand_calibration", "core.utils"]
    core_failed = []
    
    print(f"\nChecking core module imports...")
    for module in core_modules:
        try:
            __import__(module)
            print(f"PASS: {module} - Importable")
        except ImportError as e:
            print(f"FAIL: {module} - Import Error: {e}")
            core_failed.append(module)
    
    # Overall assessment
    total_issues = len(failed_packages) + len(core_failed)
    if total_issues == 0:
        print(f"\n[PASS] Environment Check: ALL GOOD - No issues found")
        return True
    else:
        print(f"\n[WARN] Environment Check: {total_issues} issues found")
        if failed_packages:
            print(f"   - Missing packages: {', '.join(failed_packages)}")
        if core_failed:
            print(f"   - Failed core imports: {', '.join(core_failed)}")
        return False

def run_unit_tests():
    """Run unit tests."""
    print("\n" + "="*60)
    print("Unit Tests")
    print("="*60)
    
    # Discover unit tests in tests directory and root
    test_patterns = [
        "test_*.py",           # Root level tests (for backward compatibility)
        "tests/unit/test_*.py", # New unit tests location
        "tests/**/test_*.py"   # Any tests in tests subdirectories
    ]
    
    test_files = set()  # Use set to avoid duplicates
    for pattern in test_patterns:
        found_files = glob.glob(pattern, recursive=True)
        # Normalize paths to avoid duplicates with different separators
        for f in found_files:
            normalized_path = os.path.normpath(f)
            test_files.add(normalized_path)
    
    # Exclude test_runner.py itself
    test_files = [f for f in test_files if not os.path.basename(f) == "test_runner.py"]
    test_files = sorted(list(test_files))  # Convert to sorted list
    
    if not test_files:
        print("No unit test files found")
        return False
    
    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file}")
    print("")
    
    success_count = 0
    total_count = len(test_files)
    
    for test_file in test_files:
        print(f"\nRunning {test_file}...")
        
        try:
            # Run the test file
            result = subprocess.run(
                [sys.executable, test_file], 
                capture_output=True, 
                text=True, 
                timeout=300,  # 5 minute timeout
                encoding='utf-8',
                errors='replace',
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8:replace'}
            )
            
            if result.returncode == 0:
                print(f"PASS - {test_file} - SUCCESS")
                success_count += 1
                
                # Show test summary if available
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines[-3:]:  # Show last few lines (usually contains summary)
                    if 'OK' in line or 'PASSED' in line or 'FAILED' in line or 'ERROR' in line:
                        print(f"   {line}")
            else:
                print(f"FAIL - {test_file} - FAILED")
                print(f"   Return code: {result.returncode}")
                if result.stderr:
                    # Show first few lines of error
                    error_lines = result.stderr.strip().split('\n')[:3]
                    for line in error_lines:
                        print(f"   Error: {line}")
                        
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT - {test_file} - TIMEOUT (5 minutes)")
            
        except Exception as e:
            print(f"ERROR - {test_file} - EXCEPTION: {e}")
    
    print(f"\nUnit Tests Summary: {success_count}/{total_count} passed")
    return success_count == total_count

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Camera Calibration Toolkit - Test Runner")
    parser.add_argument("--examples", action="store_true", help="Validate example scripts")
    parser.add_argument("--check", action="store_true", help="Check environment")
    parser.add_argument("--tests", action="store_true", help="Run unit tests")
    parser.add_argument("--all", action="store_true", help="Run all validations (examples, tests, environment)")
    
    args = parser.parse_args()
    
    # If no specific args, show help
    if not any(vars(args).values()):
        print("Camera Calibration Toolkit - Test Runner")
        print("=" * 50)
        print("Available commands:")
        print("  --examples    Validate example scripts")
        print("  --tests       Run unit tests")
        print("  --check       Check environment and dependencies")
        print("  --all         Run all validations (examples, tests, environment)")
        print()
        print("Examples:")
        print("  python test_runner.py --examples")
        print("  python test_runner.py --tests")
        print("  python test_runner.py --all")
        print("  python test_runner.py --check")
        return 0
    
    success_count = 0
    total_count = 0
    
    # Check environment first if requested
    if args.check or args.all:
        total_count += 1
        if check_environment():
            success_count += 1
    
    # Run example validation
    if args.examples or args.all:
        total_count += 1
        if validate_examples():
            success_count += 1
    
    # Run unit tests
    if args.tests or args.all:
        total_count += 1
        if run_unit_tests():
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"Successful: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("All validations passed!")
        return 0
    else:
        print("WARNING: Some validations failed or encountered issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
