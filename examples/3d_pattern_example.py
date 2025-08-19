"""
3D Calibration Pattern Example
==============================

This example demonstrates how to create and use 3D calibration patterns
where the calibration features are not on a flat plane but have known
3D coordinates in space.

Examples include:
- Markers attached to a 3D object with known geometry
- Multiple planar patterns at different orientations
- Custom 3D calibration rigs

Usage:
    conda activate camcalib
    python examples/3d_pattern_example.py
"""

import sys
import os
import numpy as np
import cv2

# Add the toolkit to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration_patterns import (
    CalibrationPattern,
    # Custom3DPattern,  # TODO: Not yet implemented in modular system
    CalibrationPatternManager,
    create_chessboard_pattern
)
from core.intrinsic_calibration import IntrinsicCalibrator


def create_cubic_calibration_object():
    """Create a cubic calibration object with markers on multiple faces."""
    
    # Define a 50mm cube with markers on three visible faces
    cube_size = 0.050  # 50mm cube
    marker_spacing = 0.010  # 10mm spacing between markers
    
    # Create 3D points for markers on three faces of the cube
    points_3d = []
    
    # Face 1: Front face (z = 0)
    for i in range(4):  # 4 markers on front face
        for j in range(4):
            x = i * marker_spacing
            y = j * marker_spacing 
            z = 0.0
            points_3d.append([x, y, z])
    
    # Face 2: Right face (x = cube_size)
    for i in range(4):  # 4 markers on right face
        for j in range(4):
            x = cube_size
            y = i * marker_spacing
            z = j * marker_spacing
            points_3d.append([x, y, z])
    
    # Face 3: Top face (y = cube_size) 
    for i in range(4):  # 4 markers on top face
        for j in range(4):
            x = i * marker_spacing
            y = cube_size
            z = j * marker_spacing
            points_3d.append([x, y, z])
    
    points_3d = np.array(points_3d)
    
    print(f"Created cubic calibration object:")
    print(f"  - Cube size: {cube_size*1000:.1f}mm")
    print(f"  - Marker spacing: {marker_spacing*1000:.1f}mm")
    print(f"  - Total markers: {len(points_3d)}")
    print(f"  - 3D coordinate range:")
    print(f"    X: [{points_3d[:,0].min()*1000:.1f}, {points_3d[:,0].max()*1000:.1f}] mm")
    print(f"    Y: [{points_3d[:,1].min()*1000:.1f}, {points_3d[:,1].max()*1000:.1f}] mm") 
    print(f"    Z: [{points_3d[:,2].min()*1000:.1f}, {points_3d[:,2].max()*1000:.1f}] mm")
    
    return points_3d


def create_pyramid_calibration_object():
    """Create a pyramid calibration object with markers on the faces."""
    
    # Define a pyramid with square base and apex
    base_size = 0.060  # 60mm base
    height = 0.040     # 40mm height
    marker_spacing = 0.015  # 15mm spacing
    
    points_3d = []
    
    # Base face (z = 0)
    base_points = int(base_size / marker_spacing) + 1
    for i in range(base_points):
        for j in range(base_points):
            x = i * marker_spacing - base_size/2
            y = j * marker_spacing - base_size/2
            z = 0.0
            points_3d.append([x, y, z])
    
    # Four triangular faces - simplified with edge markers
    apex = [0, 0, height]
    
    # Add markers along edges from corners to apex
    corners = [
        [-base_size/2, -base_size/2, 0],
        [base_size/2, -base_size/2, 0], 
        [base_size/2, base_size/2, 0],
        [-base_size/2, base_size/2, 0]
    ]
    
    for corner in corners:
        # Add 3 intermediate points between corner and apex
        for t in np.linspace(0.25, 0.75, 3):
            point = np.array(corner) * (1-t) + np.array(apex) * t
            points_3d.append(point.tolist())
    
    points_3d = np.array(points_3d)
    
    print(f"Created pyramid calibration object:")
    print(f"  - Base size: {base_size*1000:.1f}mm")
    print(f"  - Height: {height*1000:.1f}mm")
    print(f"  - Total markers: {len(points_3d)}")
    
    return points_3d


def dummy_feature_detector(image, pattern_points, **kwargs):
    """
    Dummy feature detector for demonstration purposes.
    In reality, this would implement actual feature detection (e.g., blob detection,
    template matching, deep learning-based detection, etc.)
    """
    # This is a placeholder that simulates successful detection
    # In practice, you would implement actual 2D feature detection here
    
    num_points = len(pattern_points)
    height, width = image.shape[:2] if len(image.shape) == 3 else image.shape
    
    # Simulate detecting a subset of features (70% success rate)
    num_detected = int(num_points * 0.7)
    detected_ids = np.random.choice(num_points, num_detected, replace=False)
    
    # Generate random image coordinates for detected features
    image_points = []
    for _ in range(num_detected):
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)
        image_points.append([x, y])
    
    image_points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
    detected_ids = np.sort(detected_ids)
    
    print(f"  [Simulated] Detected {num_detected}/{num_points} features")
    
    return True, image_points, detected_ids


def demonstrate_3d_patterns():
    """Demonstrate different 3D calibration patterns."""
    
    print("🎲 3D Calibration Pattern Examples")
    print("="*50)
    print()
    
    # Example 1: Cubic calibration object
    print("1. Cubic Calibration Object:")
    cube_points = create_cubic_calibration_object()
    
    cube_pattern = Custom3DPattern(
        pattern_id="cube_markers",
        name="Cubic Calibration Object",
        object_points_3d=cube_points,
        feature_detector=lambda img, **kwargs: dummy_feature_detector(img, cube_points, **kwargs)
    )
    
    print(f"   Pattern info: {cube_pattern.get_info()}")
    print()
    
    # Example 2: Pyramid calibration object 
    print("2. Pyramid Calibration Object:")
    pyramid_points = create_pyramid_calibration_object()
    
    pyramid_pattern = Custom3DPattern(
        pattern_id="pyramid_markers", 
        name="Pyramid Calibration Object",
        object_points_3d=pyramid_points,
        feature_detector=lambda img, **kwargs: dummy_feature_detector(img, pyramid_points, **kwargs)
    )
    
    print(f"   Pattern info: {pyramid_pattern.get_info()}")
    print()
    
    # Example 3: Compare with 2D planar pattern
    print("3. Comparison with 2D Planar Pattern:")
    planar_pattern = create_chessboard_pattern(
        "standard", width=11, height=8, square_size=0.020
    )
    print(f"   Pattern info: {planar_pattern.get_info()}")
    print()
    
    return cube_pattern, pyramid_pattern, planar_pattern


def simulate_3d_calibration():
    """Simulate calibration with 3D patterns."""
    
    print("🎯 Simulated 3D Calibration")
    print("="*40)
    
    # Create a 3D calibration pattern
    cube_points = create_cubic_calibration_object()
    cube_pattern = Custom3DPattern(
        pattern_id="cube_markers",
        name="Cubic Calibration Object", 
        object_points_3d=cube_points,
        feature_detector=lambda img, **kwargs: dummy_feature_detector(img, cube_points, **kwargs)
    )
    
    # Simulate some test images (using dummy images)
    print("\nSimulating calibration with synthetic images...")
    
    # Create dummy images
    dummy_images = []
    for i in range(5):
        # Create a dummy grayscale image
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        dummy_images.append(img)
    
    print(f"Created {len(dummy_images)} synthetic test images")
    
    # Test feature detection
    print("\nTesting feature detection on synthetic images:")
    total_detections = 0
    
    for i, img in enumerate(dummy_images):
        success, image_points, point_ids = cube_pattern.detect_corners(img)
        if success:
            print(f"  Image {i+1}: Detected {len(point_ids)} features")
            total_detections += len(point_ids)
            
            # Test object point generation
            obj_points = cube_pattern.generate_object_points(point_ids)
            print(f"    Generated {len(obj_points)} corresponding 3D points")
        else:
            print(f"  Image {i+1}: Detection failed")
    
    print(f"\nTotal feature detections: {total_detections}")
    print(f"Average detections per image: {total_detections/len(dummy_images):.1f}")


def demonstrate_mixed_calibration():
    """Demonstrate mixed 2D/3D calibration scenarios."""
    
    print("🔄 Mixed 2D/3D Calibration Scenarios")
    print("="*40)
    
    manager = CalibrationPatternManager()
    print(f"Available pattern types: {manager.get_available_patterns()}")
    
    scenarios = [
        {
            'name': 'Traditional 2D Planar',
            'description': 'Standard chessboard on flat surface',
            'pattern_type': 'standard',
            'is_3d': False
        },
        {
            'name': 'ChArUco 2D Planar',
            'description': 'ChArUco board on flat surface', 
            'pattern_type': 'charuco',
            'is_3d': False
        },
        {
            'name': 'Custom 3D Object',
            'description': 'Markers on 3D calibration object',
            'pattern_type': 'custom3d',
            'is_3d': True
        }
    ]
    
    print("\nCalibration scenarios:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}: {scenario['description']}")
        print(f"   3D Pattern: {'Yes' if scenario['is_3d'] else 'No'}")
    
    print("\nKey differences in 3D patterns:")
    print("✅ Advantages:")
    print("  - More robust to partial occlusions")
    print("  - Better constraint geometry")
    print("  - Can calibrate with fewer images")
    print("  - Suitable for tight spaces")
    
    print("⚠️  Challenges:")
    print("  - Requires precise 3D coordinates")
    print("  - More complex feature detection")
    print("  - Manufacturing tolerances affect accuracy")
    print("  - Custom detection algorithms needed")


def main():
    """Main demonstration function."""
    
    print("🌐 3D Calibration Pattern System Demo")
    print("="*60)
    print()
    
    # Demonstrate different 3D pattern types
    cube_pattern, pyramid_pattern, planar_pattern = demonstrate_3d_patterns()
    
    # Simulate 3D calibration process
    simulate_3d_calibration()
    
    # Show mixed calibration scenarios
    demonstrate_mixed_calibration()
    
    print("\n🎉 3D Pattern Demo Completed!")
    print("\nNext Steps for Real Implementation:")
    print("1. Implement actual feature detection for your 3D pattern")
    print("2. Measure precise 3D coordinates of calibration features")
    print("3. Validate detection accuracy and repeatability")
    print("4. Test calibration accuracy vs traditional 2D methods")
    print("5. Consider manufacturing tolerances in 3D object creation")


if __name__ == "__main__":
    main()
