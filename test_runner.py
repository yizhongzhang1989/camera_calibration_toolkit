#!/usr/bin/env python3
"""
Test Runner for Camera Calibration Toolkit

Simple script to run tests with common configurations.
Provides easy commands for different test scenarios.
"""
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} - SUCCESS")
    else:
        print(f"❌ {description} - FAILED")
        return False
    return True

def run_unit_tests():
    """Run unit tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/unit/test_patterns_simple.py", 
        "tests/unit/test_utils_simple.py", 
        "-v"
    ]
    return run_command(cmd, "Unit Tests")

def run_tests_with_coverage():
    """Run tests with coverage report."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/unit/test_patterns_simple.py", 
        "tests/unit/test_utils_simple.py", 
        "--cov=core", 
        "--cov-report=term-missing",
        "-v"
    ]
    return run_command(cmd, "Tests with Coverage")

def run_all_tests():
    """Run all available tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/unit/", 
        "-v", 
        "--tb=short"
    ]
    return run_command(cmd, "All Available Tests")

def run_integration_tests():
    """Run integration tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/integration/", 
        "-v",
        "-m", "integration"
    ]
    return run_command(cmd, "Integration Tests")

def run_e2e_tests():
    """Run end-to-end tests."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/e2e/", 
        "-v",
        "-m", "e2e"
    ]
    return run_command(cmd, "End-to-End Tests")

def validate_examples():
    """Validate example scripts."""
    examples = [
        "examples/intrinsic_calibration_example.py",
        "examples/eye_in_hand_calibration_example.py"
    ]
    
    success = True
    for example in examples:
        if Path(example).exists():
            cmd = [sys.executable, example, "--validate"]
            if not run_command(cmd, f"Validate {example}"):
                success = False
        else:
            print(f"⚠️  Example not found: {example}")
    
    return success

def check_test_environment():
    """Check test environment setup."""
    print("\n" + "="*60)
    print("🔍 Testing Environment Check")
    print("="*60)
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    
    # Read requirements.txt to get all required packages
    requirements_file = Path("requirements.txt")
    required_packages = ["pytest", "pytest-cov"]  # Testing essentials
    
    if requirements_file.exists():
        print(f"📋 Reading requirements from: {requirements_file}")
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before >= or == or other version specs)
                    package_name = line.split('>=')[0].split('==')[0].split('<')[0].strip()
                    if package_name:
                        required_packages.append(package_name)
    else:
        print("⚠️  requirements.txt not found, checking essential packages only")
        # Add essential packages for testing
        required_packages.extend(["numpy", "opencv-contrib-python"])
    
    print(f"🔍 Checking {len(required_packages)} packages...")
    failed_packages = []
    
    # Check required packages
    for package in required_packages:
        try:
            # Handle special package name mappings
            import_name = package.replace("-", "_")
            if package == "opencv-contrib-python":
                import_name = "cv2"
            elif package == "PyYAML":
                import_name = "yaml"
            elif package == "Pillow":
                import_name = "PIL"
            
            __import__(import_name)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Missing")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n📋 To install missing packages:")
        print(f"pip install {' '.join(failed_packages)}")
        return False
    
    # Check test structure
    test_paths = ["tests/unit/", "tests/integration/", "tests/e2e/"]
    for path in test_paths:
        if Path(path).exists():
            print(f"✅ {path} - Available")
        else:
            print(f"❌ {path} - Missing")
    
    return True

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Camera Calibration Toolkit Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--all", action="store_true", help="Run all available tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests")
    parser.add_argument("--examples", action="store_true", help="Validate example scripts")
    parser.add_argument("--check", action="store_true", help="Check test environment")
    parser.add_argument("--quick", action="store_true", help="Run quick test suite (unit + coverage)")
    
    args = parser.parse_args()
    
    # If no specific args, show help
    if not any(vars(args).values()):
        print("Camera Calibration Toolkit - Test Runner")
        print("=========================================")
        print()
        print("Available commands:")
        print("  --unit        Run unit tests")
        print("  --coverage    Run tests with coverage")  
        print("  --all         Run all available tests")
        print("  --integration Run integration tests")
        print("  --e2e         Run end-to-end tests")
        print("  --examples    Validate example scripts")
        print("  --check       Check test environment")
        print("  --quick       Run quick test suite")
        print()
        print("Examples:")
        print("  python test_runner.py --quick")
        print("  python test_runner.py --unit --coverage")
        print("  python test_runner.py --all")
        return
    
    success_count = 0
    total_count = 0
    
    # Check environment first if requested
    if args.check:
        total_count += 1
        if check_test_environment():
            success_count += 1
    
    # Run requested tests
    if args.unit or args.quick:
        total_count += 1
        if run_unit_tests():
            success_count += 1
    
    if args.coverage or args.quick:
        total_count += 1
        if run_tests_with_coverage():
            success_count += 1
    
    if args.all:
        total_count += 1
        if run_all_tests():
            success_count += 1
    
    if args.integration:
        total_count += 1
        if run_integration_tests():
            success_count += 1
    
    if args.e2e:
        total_count += 1
        if run_e2e_tests():
            success_count += 1
    
    if args.examples:
        total_count += 1
        if validate_examples():
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Test Summary")
    print(f"{'='*60}")
    print(f"Successful: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed or encountered issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
