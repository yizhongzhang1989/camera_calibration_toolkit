# Camera Calibration Toolkit Examples

This directory contains example scripts demonstrating how to use the Camera Calibration Toolkit for various calibration tasks. All examples are automatically tested as part of the CI/CD pipeline to ensure they remain functional and up-to-date with the latest API changes.

## 📋 Available Examples

### Core Calibration Examples

1. **`intrinsic_calibration_example.py`** - Single camera intrinsic parameter calibration
   - Demonstrates three pattern types: Chessboard, ChArUco, GridBoard
   - Shows both image_paths and direct image loading approaches
   - Includes RMS error validation (0.5 pixel threshold)
   - Automatic pattern detection and result visualization

2. **`eye_in_hand_calibration_example.py`** - Robot-mounted camera calibration
   - Hand-eye calibration for cameras mounted on robot end-effector
   - Combines intrinsic and extrinsic calibration
   - Demonstrates pose data integration

3. **`eye_to_hand_calibration_example.py`** - Fixed camera calibration
   - Hand-eye calibration for stationary cameras observing robot motion
   - Shows transformation between camera and robot base coordinates

### Utility Examples

4. **`generate_chessboard_images.py`** - Synthetic calibration pattern generation
   - Creates chessboard patterns with various configurations
   - Generates pattern images and JSON configuration files
   - Useful for testing and validation purposes

## 🔄 Automatic Testing System

### How It Works

All example files in this directory are automatically validated through:

1. **GitHub Actions CI/CD Pipeline**
   - Runs on every push and pull request
   - Tests across multiple Python versions (3.8-3.11)
   - Tests on both Ubuntu and Windows platforms

2. **Local Test Runner**
   ```bash
   python test_runner.py --examples
   ```

3. **Quick Test Suite**
   ```bash
   python test_runner.py --quick
   ```

### Test Criteria

Each example must:
- ✅ **Execute successfully** (return code 0)
- ✅ **Complete within 5 minutes** (timeout protection)
- ✅ **Handle errors gracefully** (proper exception handling)
- ✅ **Use proper exit codes** (0 for success, non-zero for failure)
- ✅ **Be self-contained** (not depend on external files beyond sample_data/)

### Test Environment

- **Unicode Safety**: Automatic encoding handling for cross-platform compatibility
- **Resource Management**: Proper cleanup of temporary files and directories
- **Error Reporting**: Detailed error messages and debugging information
- **Sample Data**: Access to `sample_data/` directory with test images and configurations

## 📝 Guidelines for Adding New Examples

### 1. File Structure and Naming

```python
#!/usr/bin/env python3
"""
[Example Name] Example
=====================

Brief description of what this example demonstrates.
Key concepts and use cases covered.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Your imports here
from core.module_name import ClassName
```

### 2. Required Components

#### Main Function with Error Handling
```python
def main():
    """Main function with proper error handling."""
    try:
        # Your example logic here
        print("✅ Example completed successfully")
        return 0
    except Exception as e:
        print(f"❌ Example failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
```

#### Progress Indicators
```python
print("🔧 Starting calibration process...")
print("=" * 50)
print("✅ Pattern detection completed")
print("🔍 Generating visualization...")
```

#### Error Messages with Debugging Tips
```python
if not success:
    print("❌ Calibration failed!")
    print("   Check that:")
    print("   - Sample data is available")
    print("   - Pattern configuration is correct")
    print("   - Images are not corrupted")
```

### 3. Documentation Requirements

#### File Header
- Clear title and description
- List of key concepts demonstrated
- Any special requirements or dependencies

#### Inline Comments
```python
# Smart constructor - sets member parameters directly
calibrator = IntrinsicCalibrator(
    image_paths=image_paths,           # Member parameter set in constructor
    calibration_pattern=pattern       # Member parameter set in constructor
)
```

#### Result Reporting
```python
print(f"📊 Results: {success_count}/{total_tests} operations successful")
if success_count == total_tests:
    print("All operations completed successfully!")
    return 0
else:
    print("⚠️ Some operations failed. Check error messages above.")
    return 1
```

### 4. Sample Data Usage

Examples should use the provided sample data:
```python
# Standard sample data directories
sample_dir = os.path.join("sample_data", "eye_in_hand_test_data")
charuco_dir = os.path.join("sample_data", "intrinsic_calib_charuco_test_images")
grid_dir = os.path.join("sample_data", "intrinsic_calib_grid_test_images")

# Always check if sample data exists
if not os.path.exists(sample_dir):
    error_msg = f"Sample data directory not found: {sample_dir}"
    print(f"❌ {error_msg}")
    raise FileNotFoundError(error_msg)
```

### 5. Output Management

#### Results Directory Structure
```python
# Create organized output directories
output_dir = f"data/results/{pattern.pattern_id}_calibration"
os.makedirs(output_dir, exist_ok=True)

# Save calibration results
calibrator.save_calibration(
    os.path.join(output_dir, "calibration_results.json"),
    include_extrinsics=True
)

# Generate debug images
pattern_debug_dir = os.path.join(output_dir, "pattern_detection")
os.makedirs(pattern_debug_dir, exist_ok=True)
```

### 6. API Best Practices

#### Use Modern API Design
```python
# ✅ Good - Modern boolean API
success = calibrator.calibrate(verbose=True)
if success:
    rms_error = calibrator.get_rms_error()
    camera_matrix = calibrator.get_camera_matrix()

# ❌ Avoid - Old RMS-based API
rms_error = calibrator.calibrate_camera(verbose=True)
if rms_error > 0:  # Confusing semantics
```

#### Include Quality Validation
```python
# Check calibration quality
if rms_error > 0.5:  # or appropriate threshold
    print(f"❌ Calibration failed - RMS error too high: {rms_error:.4f}")
    return 1
```

### 7. Testing Your Example

Before submitting, test your example:

```bash
# Test individual example
python examples/your_new_example.py

# Test with the validation system
python test_runner.py --examples

# Run full test suite
python test_runner.py --quick
```

### 8. Common Patterns to Follow

#### Multiple Test Functions
```python
def test_feature_a():
    """Test specific feature A."""
    # Implementation here
    
def test_feature_b():
    """Test specific feature B."""
    # Implementation here

def main():
    """Main function coordinating all tests."""
    success_count = 0
    total_tests = 2
    
    try:
        test_feature_a()
        success_count += 1
    except Exception as e:
        print(f"❌ Feature A failed: {e}")
    
    try:
        test_feature_b()
        success_count += 1
    except Exception as e:
        print(f"❌ Feature B failed: {e}")
    
    return 0 if success_count == total_tests else 1
```

#### Resource Cleanup
```python
# Clean up temporary files
import tempfile
import shutil

with tempfile.TemporaryDirectory() as temp_dir:
    # Use temp_dir for temporary files
    # Automatic cleanup on exit
    pass
```

## 🚀 Integration with CI/CD

Your example will be automatically:
- **Discovered** by the test system (no manual registration needed)
- **Executed** on multiple platforms and Python versions
- **Validated** for proper exit codes and error handling
- **Reported** in CI/CD pipeline results

## 💡 Best Practices Summary

1. **Self-contained**: Don't depend on external resources
2. **Robust**: Handle missing files and invalid data gracefully
3. **Informative**: Provide clear progress indicators and error messages
4. **Efficient**: Complete within reasonable time limits
5. **Cross-platform**: Use `os.path.join()` and avoid platform-specific code
6. **Standards-compliant**: Follow PEP 8 and existing code style
7. **Well-documented**: Include docstrings and inline comments
8. **Exit codes**: Return 0 for success, non-zero for failure

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure proper path setup for core modules
2. **File Not Found**: Always check if sample data exists before using
3. **Unicode Issues**: The test system handles encoding automatically
4. **Timeout**: Keep examples under 5-minute runtime
5. **Platform Differences**: Use cross-platform file operations

### Getting Help

- Check existing examples for patterns and best practices
- Review the core module documentation
- Test locally before committing
- Monitor CI/CD pipeline results for cross-platform issues

---

*This README is automatically validated along with the examples to ensure it stays current with the codebase.*
