#!/usr/bin/env python3
"""
Pattern Serialization Example
=============================

This example demonstrates how to save and load calibration patterns using JSON.

Key Features:
- Save any calibration pattern to JSON format
- Load calibration patterns from JSON format
- Perfect for storing pattern configurations in calibration result files
- Enables pattern recovery when analyzing calibration data later
"""

import sys
import os
import json

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration_patterns import (
    get_pattern_manager, 
    save_pattern_to_json, 
    load_pattern_from_json
)


def save_patterns_example():
    """Example of saving different pattern types to JSON."""
    
    print("=== Saving Calibration Patterns to JSON ===\n")
    
    manager = get_pattern_manager()
    
    # Create different pattern types
    patterns_to_save = [
        {
            'name': 'standard_chessboard_11x8',
            'pattern': manager.create_pattern(
                'standard_chessboard', 
                width=11, height=8, square_size=0.025
            )
        },
        {
            'name': 'charuco_8x6', 
            'pattern': manager.create_pattern(
                'charuco_board',
                width=8, height=6, 
                square_size=0.03, marker_size=0.02, 
                dictionary_id=10  # DICT_6X6_250
            )
        },
        {
            'name': 'grid_board_5x4',
            'pattern': manager.create_pattern(
                'grid_board',
                markers_x=5, markers_y=4,
                marker_size=0.04, marker_separation=0.01,
                dictionary_id=10  # DICT_6X6_250
            )
        }
    ]
    
    # Save each pattern
    saved_patterns = {}
    for pattern_info in patterns_to_save:
        name = pattern_info['name']
        pattern = pattern_info['pattern']
        
        # Convert to JSON
        pattern_json = save_pattern_to_json(pattern)
        saved_patterns[name] = pattern_json
        
        print(f"Saved {name}:")
        print(f"  Pattern ID: {pattern_json['pattern_id']}")
        print(f"  Name: {pattern_json['name']}")
        print(f"  Parameters: {pattern_json['parameters']}")
        print()
    
    # Save all patterns to a file
    os.makedirs('data/results', exist_ok=True)
    with open('data/results/calibration_patterns.json', 'w') as f:
        json.dump(saved_patterns, f, indent=2)
    
    print("✅ All patterns saved to 'data/results/calibration_patterns.json'")
    print()
    
    return saved_patterns


def load_patterns_example():
    """Example of loading patterns from JSON."""
    
    print("=== Loading Calibration Patterns from JSON ===\n")
    
    # Load patterns from file
    try:
        with open('data/results/calibration_patterns.json', 'r') as f:
            saved_patterns = json.load(f)
    except FileNotFoundError:
        print("❌ Pattern file not found. Run save_patterns_example() first.")
        return
    
    # Load each pattern
    for name, pattern_json in saved_patterns.items():
        print(f"Loading {name}...")
        
        # Restore pattern from JSON
        restored_pattern = load_pattern_from_json(pattern_json)
        
        print(f"  Restored pattern: {restored_pattern.pattern_id}")
        print(f"  Name: {restored_pattern.name}")
        print(f"  Display: {restored_pattern.get_display_name()}")
        
        # Show pattern info
        info = restored_pattern.get_info()
        print(f"  Details: {info}")
        print()
    
    print("✅ All patterns loaded successfully")


def calibration_workflow_example():
    """Example of using JSON serialization in a calibration workflow."""
    
    print("=== Calibration Workflow with Pattern Serialization ===\n")
    
    manager = get_pattern_manager()
    
    # Step 1: Create calibration pattern
    print("1. Creating calibration pattern...")
    pattern = manager.create_pattern(
        'standard_chessboard',
        width=11, height=8, square_size=0.025
    )
    print(f"   Created: {pattern.get_display_name()}")
    
    # Step 2: Save pattern configuration (would be part of calibration results)
    print("\n2. Saving pattern configuration...")
    pattern_json = save_pattern_to_json(pattern)
    
    # Simulate saving calibration results
    calibration_results = {
        'timestamp': '2025-08-20T10:30:00Z',
        'calibration_pattern': pattern_json,
        'camera_matrix': [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]],  # Example data
        'distortion_coefficients': [0.1, -0.2, 0.001, 0.002, 0.0],
        'reprojection_error': 0.25
    }
    
    with open('data/results/example_calibration.json', 'w') as f:
        json.dump(calibration_results, f, indent=2)
    print("   Saved calibration results with pattern info")
    
    # Step 3: Later, load calibration results and recover the pattern
    print("\n3. Loading calibration results and recovering pattern...")
    with open('data/results/example_calibration.json', 'r') as f:
        loaded_results = json.load(f)
    
    # Recover the exact pattern used in calibration
    recovered_pattern = load_pattern_from_json(loaded_results['calibration_pattern'])
    print(f"   Recovered pattern: {recovered_pattern.get_display_name()}")
    print(f"   Pattern matches original: {pattern.pattern_id == recovered_pattern.pattern_id}")
    print(f"   Parameters match: {pattern.width == recovered_pattern.width and pattern.height == recovered_pattern.height}")
    
    print("\n✅ Calibration workflow complete with pattern recovery")


def main():
    """Run all examples with proper error handling."""
    
    print("Calibration Pattern JSON Serialization Examples")
    print("=" * 50)
    print()
    
    success_count = 0
    total_tests = 3
    
    try:
        print("Testing pattern saving...")
        saved_patterns = save_patterns_example()
        success_count += 1
        print("✅ Pattern saving completed successfully")
    except Exception as e:
        print(f"❌ Pattern saving failed: {e}")
    
    try:
        print("\nTesting pattern loading...")
        load_patterns_example()
        success_count += 1
        print("✅ Pattern loading completed successfully")
    except Exception as e:
        print(f"❌ Pattern loading failed: {e}")
    
    try:
        print("\nTesting calibration workflow...")
        calibration_workflow_example()
        success_count += 1
        print("✅ Calibration workflow completed successfully")
    except Exception as e:
        print(f"❌ Calibration workflow failed: {e}")
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("- to_json(): Convert pattern to JSON dict")
    print("- from_json(): Create pattern from JSON dict")  
    print("- save_pattern_to_json(): Convenience function")
    print("- load_pattern_from_json(): Convenience function")
    print("- Perfect for storing pattern info in calibration results")
    print("- Enables exact pattern recovery for later analysis")
    
    print(f"\n📊 Results: {success_count}/{total_tests} tests successful")
    
    if success_count == total_tests:
        return 0
    elif success_count > 0:
        print(f"⚠️  Some tests failed. Check error messages above.")
        return 1
    else:
        print(f"❌ All tests failed!")
        return 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
