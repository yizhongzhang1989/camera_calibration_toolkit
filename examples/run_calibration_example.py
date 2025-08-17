"""
Simple Camera Calibration Example
=================================

This example demonstrates basic camera calibration using the sample test images.

Features:
- Loads images from sample_data directory
- Performs intrinsic calibration with standard distortion model
- Saves calibration results in multiple formats
- Shows basic calibration metrics

Usage:
    conda activate camcalib
    python examples/run_calibration_example.py
"""

import sys
import os
import glob
import numpy as np
import cv2
import json
from datetime import datetime

# Add the toolkit to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intrinsic_calibration import IntrinsicCalibrator


def load_sample_images(sample_data_dir=None):
    """Load images from sample_data directory."""
    if sample_data_dir is None:
        toolkit_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sample_data_dir = os.path.join(toolkit_root, "sample_data", "intrinsic_calib_test_images")
    
    if not os.path.exists(sample_data_dir):
        raise FileNotFoundError(f"Sample data directory not found: {sample_data_dir}")
    
    # Load all common image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(sample_data_dir, extension)))
        image_paths.extend(glob.glob(os.path.join(sample_data_dir, extension.upper())))
    
    # Remove duplicates (in case same files are matched by different patterns)
    image_paths = list(set(image_paths))
    
    # Sort numerically if possible
    try:
        image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except ValueError:
        image_paths.sort()
    
    print(f"Found {len(image_paths)} images")
    return image_paths
    return image_paths


def save_results(results, output_dir="results"):
    """Save calibration results."""
    if not results or not results.get('success'):
        print("No successful calibration results to save")
        return None
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = os.path.join(output_dir, f"intrinsic_calibration_{timestamp}")
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Save in JSON format
    json_path = os.path.join(full_output_dir, "calibration_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save camera matrix and distortion coefficients
    np.savetxt(os.path.join(full_output_dir, "camera_matrix.txt"), 
               np.array(results['camera_matrix']))
    np.savetxt(os.path.join(full_output_dir, "distortion_coefficients.txt"), 
               np.array(results['distortion_coefficients']))
    
    # Create a summary
    summary_path = os.path.join(full_output_dir, "calibration_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Camera Intrinsic Calibration Results\n")
        f.write("="*40 + "\n\n")
        f.write(f"Calibration Date: {results['calibration_timestamp']}\n")
        f.write(f"Chessboard Pattern: {results['parameters']['chessboard_pattern']}\n")
        f.write(f"Square Size: {results['parameters']['square_size_meters']}m\n")
        f.write(f"Images Used: {results['parameters']['valid_images_used']}/{results['parameters']['total_input_images']}\n\n")
        
        cam_matrix = np.array(results['camera_matrix'])
        f.write("Camera Matrix:\n")
        f.write(f"  fx: {cam_matrix[0,0]:.2f}  fy: {cam_matrix[1,1]:.2f}\n")
        f.write(f"  cx: {cam_matrix[0,2]:.2f}  cy: {cam_matrix[1,2]:.2f}\n\n")
        f.write(f"Distortion Coefficients: {np.array(results['distortion_coefficients']).flatten()}\n")
    
    print(f"Results saved to: {full_output_dir}")
    return full_output_dir


def main():
    """Simple camera calibration example."""
    
    print("📷 Simple Camera Calibration Example")
    print("="*50)
    print()
    
    # Configuration
    chessboard_x = 11      # Chessboard corners along x-axis
    chessboard_y = 8       # Chessboard corners along y-axis 
    square_size = 0.02     # Size of each chessboard square in meters
    distortion_model = 'standard'  # Distortion model
    
    print("📋 Configuration:")
    print(f"   Chessboard pattern: {chessboard_x}x{chessboard_y}")
    print(f"   Square size: {square_size}m")
    print(f"   Distortion model: {distortion_model}")
    print()
    
    try:
        # Step 1: Load images
        print("Step 1: Loading sample images...")
        image_paths = load_sample_images()
        
        if not image_paths:
            raise ValueError("No images found in sample data directory")
        
        # Step 2: Calibrate
        print(f"\nStep 2: Running calibration with {len(image_paths)} images...")
        
        # Initialize calibrator
        calibrator = IntrinsicCalibrator()
        
        # Perform calibration
        success, camera_matrix, distortion_coeffs = calibrator.calibrate_from_images(
            image_paths=image_paths,
            XX=chessboard_x,
            YY=chessboard_y,
            L=square_size,
            distortion_model=distortion_model,
            verbose=True
        )
        
        if success:
            print("✅ Calibration completed successfully!")
            
            # Display key results
            print("\nCamera Matrix:")
            print(f"  fx: {camera_matrix[0,0]:.2f}, fy: {camera_matrix[1,1]:.2f}")
            print(f"  cx: {camera_matrix[0,2]:.2f}, cy: {camera_matrix[1,2]:.2f}")
            print(f"Distortion Coefficients: {distortion_coeffs.flatten()}")
            
            # Prepare results for saving
            results = {
                'success': True,
                'calibration_timestamp': datetime.now().isoformat(),
                'parameters': {
                    'chessboard_pattern': f"{chessboard_x}x{chessboard_y}",
                    'square_size_meters': square_size,
                    'distortion_model': distortion_model,
                    'total_input_images': len(image_paths),
                    'valid_images_used': len(calibrator.valid_image_paths) if calibrator.valid_image_paths else 0
                },
                'camera_matrix': camera_matrix.tolist(),
                'distortion_coefficients': distortion_coeffs.tolist(),
                'image_size': list(calibrator.image_size) if calibrator.image_size else None
            }
            
            # Step 3: Save results
            print("\nStep 3: Saving results...")
            output_dir = save_results(results)
            print(f"📁 Results saved to: {output_dir}")
            
            print(f"\n🎉 Calibration completed successfully!")
            print(f"Check the output directory for detailed results.")
            
        else:
            print("❌ Calibration failed!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print(f"Please check that sample images exist in sample_data/intrinsic_calib_test_images/")


if __name__ == "__main__":
    main()
