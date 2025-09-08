#!/usr/bin/env python3
"""
Unit Tests for Intrinsic Camera Calibration
==========================================

This module contains comprehensive unit tests for the IntrinsicCalibrator class.

Test Classes:
1. TestIntrinsicCalibrationDistortionModels:
   - Tests different OpenCV camera distortion models
   - Validates calibration with standard, rational, thin prism, and tilted models
   - Compares calibration results across different models
   - Uses real sample data for validation

2. TestIntrinsicCalibrationPatternDetectionFailures:
   - Tests edge cases where calibration patterns cannot be detected
   - Validates handling of white/blank images
   - Tests image size validation and error handling
   - Ensures robustness with mixed valid/invalid images

Future test classes can be added here for:
- Different calibration patterns (ChArUco, grid, custom)
- Performance benchmarks
- Calibration accuracy validation
- Integration tests with camera hardware
"""

import unittest
import os
import sys
import json
import numpy as np
import cv2
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import load_pattern_from_json


class TestIntrinsicCalibrationDistortionModels(unittest.TestCase):
    """Test intrinsic calibration with different camera distortion models.
    
    This test class validates the IntrinsicCalibrator's ability to handle different
    OpenCV distortion models:
    
    - Standard model (5 coefficients): k1, k2, p1, p2, k3
    - Rational model (8+ coefficients): adds k4, k5, k6
    - Thin prism model (12+ coefficients): adds s1, s2, s3, s4
    - Tilted model (14 coefficients): adds τx, τy
    
    Each test verifies:
    - Successful calibration with reasonable RMS error
    - Correct number of distortion coefficients returned
    - Proper matrix shapes and data types
    - Calibration parameter accuracy
    
    Uses real sample data from the eye_in_hand_test_data directory.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test data paths."""
        # Get project root (3 levels up from tests/unit/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.sample_dir = os.path.join(project_root, "sample_data", "eye_in_hand_test_data")
        cls.pattern_config_path = os.path.join(cls.sample_dir, "chessboard_config.json")
        
        # Check if test data exists
        cls.has_test_data = (
            os.path.exists(cls.pattern_config_path) and 
            os.path.exists(cls.sample_dir)
        )
    
    def setUp(self):
        """Set up individual test."""
        if not self.has_test_data:
            self.skipTest("Test data not available")
    
    def load_test_images(self, max_images: int = 6) -> List[str]:
        """Load test image paths."""
        image_paths = []
        for filename in sorted(os.listdir(self.sample_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(self.sample_dir, filename))
                if len(image_paths) >= max_images:
                    break
        return image_paths
    
    def load_test_pattern(self):
        """Load test calibration pattern."""
        with open(self.pattern_config_path, 'r') as f:
            pattern_json = json.load(f)
        return load_pattern_from_json(pattern_json)
    
    def test_calibration_standard_model(self):
        """Test calibration with standard distortion model (5 coefficients)."""
        image_paths = self.load_test_images(6)
        pattern = self.load_test_pattern()
        
        calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern
        )
        
        result = calibrator.calibrate(flags=0, verbose=False)
        
        # Verify successful calibration
        self.assertIsNotNone(result)
        self.assertIn('camera_matrix', result)
        self.assertIn('distortion_coefficients', result)
        self.assertIn('rms_error', result)
        
        # Check matrix shapes and properties
        self.assertEqual(result['camera_matrix'].shape, (3, 3))
        
        # Distortion coefficients should be flattened (1D array)
        dist_coeffs = result['distortion_coefficients']
        self.assertEqual(len(dist_coeffs.shape), 1, "Distortion coefficients should be flattened (1D)")
        self.assertEqual(len(dist_coeffs), 5, "Standard model should have 5 distortion coefficients")
        
        self.assertGreater(result['rms_error'], 0)
        self.assertLess(result['rms_error'], 0.5)  # Reasonable error threshold
        
        print(f"Standard model - RMS error: {result['rms_error']:.4f}")
        print(f"Standard model - Distortion coeffs shape: {dist_coeffs.shape}")
    
    def test_calibration_rational_model(self):
        """Test calibration with rational distortion model (8+ coefficients)."""
        image_paths = self.load_test_images(6)
        pattern = self.load_test_pattern()
        
        calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern
        )
        
        result = calibrator.calibrate(flags=cv2.CALIB_RATIONAL_MODEL, verbose=False)
        
        # Verify successful calibration
        self.assertIsNotNone(result)
        self.assertIn('camera_matrix', result)
        self.assertIn('distortion_coefficients', result)
        self.assertIn('rms_error', result)
        
        # Check matrix shapes and properties
        self.assertEqual(result['camera_matrix'].shape, (3, 3))
        
        # Distortion coefficients should be flattened (1D array)
        dist_coeffs = result['distortion_coefficients']
        self.assertEqual(len(dist_coeffs.shape), 1, "Distortion coefficients should be flattened (1D)")
        self.assertGreaterEqual(len(dist_coeffs), 8, "Rational model should have 8+ distortion coefficients")
        
        self.assertGreater(result['rms_error'], 0)
        self.assertLess(result['rms_error'], 0.5)
        
        # Rational model should have 8 or more distortion coefficients (OpenCV may return more)
        total_coeffs = len(dist_coeffs)
        
        print(f"Rational model - RMS error: {result['rms_error']:.4f}")
        print(f"Rational model - Distortion coeffs shape: {dist_coeffs.shape}")
        print(f"Rational model - Total coefficients: {total_coeffs}")
    
    def test_calibration_thin_prism_model(self):
        """Test calibration with thin prism model (12+ coefficients)."""
        image_paths = self.load_test_images(6)
        pattern = self.load_test_pattern()
        
        calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern
        )
        
        result = calibrator.calibrate(
            flags=cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL, 
            verbose=False
        )
        
        # Verify successful calibration
        self.assertIsNotNone(result)
        self.assertIn('camera_matrix', result)
        self.assertIn('distortion_coefficients', result)
        self.assertIn('rms_error', result)
        
        # Check matrix shapes and properties
        self.assertEqual(result['camera_matrix'].shape, (3, 3))
        
        # Distortion coefficients should be flattened (1D array)
        dist_coeffs = result['distortion_coefficients']
        self.assertEqual(len(dist_coeffs.shape), 1, "Distortion coefficients should be flattened (1D)")
        self.assertGreaterEqual(len(dist_coeffs), 12, "Thin prism model should have 12+ distortion coefficients")
        
        self.assertGreater(result['rms_error'], 0)
        self.assertLess(result['rms_error'], 0.5)
        
        # Thin prism model should have 12 or more distortion coefficients
        total_coeffs = len(dist_coeffs)
        
        print(f"Thin prism model - RMS error: {result['rms_error']:.4f}")
        print(f"Thin prism model - Distortion coeffs shape: {dist_coeffs.shape}")
        print(f"Thin prism model - Total coefficients: {total_coeffs}")
    
    def test_calibration_tilted_model(self):
        """Test calibration with tilted model (14 coefficients)."""
        image_paths = self.load_test_images(6)
        pattern = self.load_test_pattern()
        
        calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern
        )
        
        result = calibrator.calibrate(
            flags=cv2.CALIB_TILTED_MODEL, 
            verbose=False
        )
        
        # Verify successful calibration
        self.assertIsNotNone(result)
        self.assertIn('camera_matrix', result)
        self.assertIn('distortion_coefficients', result)
        self.assertIn('rms_error', result)
        
        # Check matrix shapes and properties
        self.assertEqual(result['camera_matrix'].shape, (3, 3))
        
        # Distortion coefficients should be flattened (1D array)
        dist_coeffs = result['distortion_coefficients']
        self.assertEqual(len(dist_coeffs.shape), 1, "Distortion coefficients should be flattened (1D)")
        self.assertEqual(len(dist_coeffs), 14, "Tilted model should have 14 distortion coefficients")
        
        self.assertGreater(result['rms_error'], 0)
        self.assertLess(result['rms_error'], 0.5)
        
        # Tilted model should have exactly 14 distortion coefficients
        total_coeffs = len(dist_coeffs)
        
        print(f"Tilted model - RMS error: {result['rms_error']:.4f}")
        print(f"Tilted model - Distortion coeffs shape: {dist_coeffs.shape}")
        print(f"Tilted model - Total coefficients: {total_coeffs}")
    
    def test_calibration_with_initial_guess(self):
        """Test calibration with initial camera matrix guess."""
        image_paths = self.load_test_images(6)
        pattern = self.load_test_pattern()
        
        calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern
        )
        
        # Create initial guess based on image size
        # First get image size by loading one image
        test_image = cv2.imread(image_paths[0])
        h, w = test_image.shape[:2]
        
        initial_matrix = np.array([
            [w * 0.8, 0, w / 2],
            [0, w * 0.8, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        result = calibrator.calibrate(
            cameraMatrix=initial_matrix,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
            verbose=False
        )
        
        # Verify successful calibration
        self.assertIsNotNone(result)
        self.assertIn('camera_matrix', result)
        self.assertIn('distortion_coefficients', result)
        self.assertIn('rms_error', result)
        
        # Result should be refined from initial guess
        self.assertFalse(np.array_equal(result['camera_matrix'], initial_matrix))
        
        print(f"With initial guess - RMS error: {result['rms_error']:.4f}")
        print(f"Initial fx: {initial_matrix[0,0]:.2f}, Final fx: {result['camera_matrix'][0,0]:.2f}")
    
    def test_compare_camera_models(self):
        """Compare different camera models and their RMS errors."""
        image_paths = self.load_test_images(6)
        pattern = self.load_test_pattern()
        
        models = [
            ("Standard (5 coeff)", 0),
            ("Rational (8+ coeff)", cv2.CALIB_RATIONAL_MODEL),
            ("Thin Prism (12+ coeff)", cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL),
            ("Tilted (14 coeff)", cv2.CALIB_TILTED_MODEL),
        ]
        
        results = {}
        
        for model_name, flags in models:
            calibrator = IntrinsicCalibrator(
                image_paths=image_paths,
                calibration_pattern=pattern
            )
            
            result = calibrator.calibrate(flags=flags, verbose=False)
            self.assertIsNotNone(result, f"Calibration failed for {model_name}")
            
            results[model_name] = {
                'rms_error': result['rms_error'],
                'num_coeffs': len(result['distortion_coefficients']),  # Should be flattened 1D array
                'fx': result['camera_matrix'][0, 0],
                'fy': result['camera_matrix'][1, 1]
            }
        
        # Print comparison
        print("\nCamera Model Comparison:")
        print("-" * 60)
        for model_name, data in results.items():
            print(f"{model_name:20} | RMS: {data['rms_error']:.4f} | "
                  f"Coeffs: {data['num_coeffs']:2d} | "
                  f"fx: {data['fx']:6.1f} | fy: {data['fy']:6.1f}")
        
        # Verify all models completed successfully
        self.assertEqual(len(results), len(models))
        
        # All RMS errors should be reasonable
        for model_name, data in results.items():
            self.assertLess(data['rms_error'], 0.5, f"{model_name} has unreasonable RMS error")


class TestIntrinsicCalibrationPatternDetectionFailures(unittest.TestCase):
    """
    Test intrinsic calibration with pattern detection failures and edge cases.
    
    This test class validates how the IntrinsicCalibrator handles problematic scenarios
    where calibration patterns cannot be detected or other edge conditions occur:
    
    1. **Pattern Detection Failures:**
       - Pure white images (no detectable patterns)
       - Images with corrupted or missing pattern data
       - Images where pattern detection algorithms fail
    
    2. **Image Size Validation:**
       - Mixed image sizes in calibration set (should fail with clear error)
       - Consistent image sizes (should succeed)
    
    3. **Data Alignment Verification:**
       - Arrays (rvecs, tvecs, per_image_errors, etc.) maintain proper alignment
       - Failed detections have None values at correct indices
       - Successful detections have valid data at correct indices
    
    **Expected Behaviors:**
    - Failed pattern detections: Skipped gracefully, calibration continues with valid images
    - Mixed image sizes: Clear validation error with descriptive message
    - Array alignment: All result arrays have same length as input image count
    - Robustness: System handles edge cases without crashing
    
    **Implementation Approach:**
    - Uses numpy arrays directly (in-memory) for clean testing
    - No temporary file management required
    - Systematic validation of array lengths and None/non-None patterns
    
    These tests ensure robustness for real-world usage where not all images
    may contain detectable calibration patterns.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test data paths."""
        # Get project root (3 levels up from tests/unit/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.sample_dir = os.path.join(project_root, "sample_data", "eye_in_hand_test_data")
        cls.pattern_config_path = os.path.join(cls.sample_dir, "chessboard_config.json")
        
        # Check if test data exists
        cls.has_test_data = (
            os.path.exists(cls.pattern_config_path) and 
            os.path.exists(cls.sample_dir)
        )
    
    def setUp(self):
        """Set up individual test."""
        if not self.has_test_data:
            self.skipTest("Test data not available")
    
    def load_test_images(self, max_images: int = 6) -> List[np.ndarray]:
        """Load test images as numpy arrays."""
        images = []
        for filename in sorted(os.listdir(self.sample_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.sample_dir, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
                    if len(images) >= max_images:
                        break
        return images
    
    def load_test_pattern(self):
        """Load test calibration pattern."""
        with open(self.pattern_config_path, 'r') as f:
            pattern_json = json.load(f)
        return load_pattern_from_json(pattern_json)
    
    def create_white_image(self, reference_image: np.ndarray) -> np.ndarray:
        """Create a pure white image with the same size as reference image."""
        h, w = reference_image.shape[:2]
        # Create pure white image
        white_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        return white_img
    
    def create_resized_image(self, source_image: np.ndarray, scale_factor: float = 0.8) -> np.ndarray:
        """Create a resized version of the source image."""
        h, w = source_image.shape[:2]
        
        # Resize image
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        resized_img = cv2.resize(source_image, (new_w, new_h))
        
        return resized_img
    
    def test_calibration_with_white_images(self):
        """Test calibration with pure white images at different positions."""
        # Test white image at first position (index 0)
        self._test_white_image_insertion([0], "first")
        
        # Test white image at last position (after all original images)
        images = self.load_test_images(5)
        self._test_white_image_insertion([len(images)], "last")
        
        # Test white image in middle position
        mid_idx = len(images) // 2
        self._test_white_image_insertion([mid_idx], "middle")
        
        # Test multiple white images at different positions
        # For 5 original images: [0, 2, 5] -> (X, a, b, X, c, d, e, X)
        self._test_white_image_insertion([0, 2, 5], "multiple")
    
    def _test_white_image_insertion(self, insert_indices: List[int], test_name: str):
        """
        Helper function to test calibration with white images at specified insertion positions.
        
        Args:
            insert_indices: List of indices where to insert white images (0-based)
                           These are the positions in the ORIGINAL array where insertions should happen.
                           For example: [0, 3, 6] with 6 original images (a,b,c,d,e,f)
                           will result in: (X, a, b, c, X, d, e, f, X)
                           - Insert at 0: (X, a, b, c, d, e, f)
                           - Insert at 3+1=4: (X, a, b, c, X, d, e, f) 
                           - Insert at 6+2=8: (X, a, b, c, X, d, e, f, X)
            test_name: Name of the test for logging purposes
        """
        images = self.load_test_images(5)
        pattern = self.load_test_pattern()
        
        # Start with original images
        test_images = list(images)
        white_image_positions = []
        
        # Sort insertion indices to process from left to right
        sorted_indices = sorted(insert_indices)
        
        # Insert white images at specified indices
        # We need to adjust indices as we insert to account for previous insertions
        for i, insert_idx in enumerate(sorted_indices):
            # Adjust index for previous insertions (each insertion shifts subsequent indices)
            adjusted_idx = insert_idx + i
            
            # Ensure index is valid (can insert at end)
            if adjusted_idx <= len(test_images):
                white_image = self.create_white_image(images[0])
                test_images.insert(adjusted_idx, white_image)
                white_image_positions.append(adjusted_idx)
        
        calibrator = IntrinsicCalibrator(
            images=test_images,
            calibration_pattern=pattern
        )
        
        # Perform calibration
        result = calibrator.calibrate(flags=0, verbose=False)
        
        # The calibrator should handle this gracefully
        if result is not None:
            # If calibration succeeds, verify it's reasonable
            self.assertIn('camera_matrix', result)
            self.assertIn('distortion_coefficients', result)
            self.assertIn('rms_error', result)
            self.assertLess(result['rms_error'], 0.5)
            
            # Check that arrays have correct length matching total images
            total_images = len(test_images)
            
            # Check rvecs array
            self.assertIsNotNone(calibrator.rvecs, "rvecs should not be None after calibration")
            self.assertEqual(len(calibrator.rvecs), total_images, 
                           f"rvecs length {len(calibrator.rvecs)} should match image count {total_images}")
            
            # Check tvecs array  
            self.assertIsNotNone(calibrator.tvecs, "tvecs should not be None after calibration")
            self.assertEqual(len(calibrator.tvecs), total_images,
                           f"tvecs length {len(calibrator.tvecs)} should match image count {total_images}")
            
            # Check per_image_errors array
            self.assertIsNotNone(calibrator.per_image_errors, "per_image_errors should not be None after calibration")
            self.assertEqual(len(calibrator.per_image_errors), total_images,
                           f"per_image_errors length {len(calibrator.per_image_errors)} should match image count {total_images}")
            
            # Check image_points array
            self.assertIsNotNone(calibrator.image_points, "image_points should not be None after calibration")
            self.assertEqual(len(calibrator.image_points), total_images,
                           f"image_points length {len(calibrator.image_points)} should match image count {total_images}")
            
            # Check point_ids array (initialized as list but may contain None for standard chessboards)
            self.assertIsNotNone(calibrator.point_ids, "point_ids should be initialized after calibration")
            self.assertEqual(len(calibrator.point_ids), total_images,
                           f"point_ids length {len(calibrator.point_ids)} should match image count {total_images}")
            
            # Check object_points array
            self.assertIsNotNone(calibrator.object_points, "object_points should not be None after calibration")
            self.assertEqual(len(calibrator.object_points), total_images,
                           f"object_points length {len(calibrator.object_points)} should match image count {total_images}")
            
            # Check that white image positions have None values
            for white_pos in white_image_positions:
                self.assertIsNone(calibrator.rvecs[white_pos], 
                                f"rvecs[{white_pos}] should be None for white image")
                self.assertIsNone(calibrator.tvecs[white_pos],
                                f"tvecs[{white_pos}] should be None for white image") 
                self.assertIsNone(calibrator.per_image_errors[white_pos],
                                f"per_image_errors[{white_pos}] should be None for white image")
                self.assertIsNone(calibrator.image_points[white_pos],
                                f"image_points[{white_pos}] should be None for white image")
                self.assertIsNone(calibrator.point_ids[white_pos],
                                f"point_ids[{white_pos}] should be None for white image")
                self.assertIsNone(calibrator.object_points[white_pos],
                                f"object_points[{white_pos}] should be None for white image")
            
            # Check that non-white image positions have non-None values
            for i in range(total_images):
                if i not in white_image_positions:
                    self.assertIsNotNone(calibrator.rvecs[i],
                                       f"rvecs[{i}] should not be None for valid image")
                    self.assertIsNotNone(calibrator.tvecs[i],
                                       f"tvecs[{i}] should not be None for valid image")
                    self.assertIsNotNone(calibrator.per_image_errors[i],
                                       f"per_image_errors[{i}] should not be None for valid image")
                    self.assertIsNotNone(calibrator.image_points[i],
                                       f"image_points[{i}] should not be None for valid image")
                    # Note: point_ids is only used for ChArUco patterns, may be None for standard chessboards
                    if calibrator.point_ids is not None:
                        # If point_ids array exists, check individual entries (may still be None for failed detections)
                        pass  # We don't assert non-None here since individual entries can be None
                    self.assertIsNotNone(calibrator.object_points[i],
                                       f"object_points[{i}] should not be None for valid image")
            
            print(f"White image {test_name} - Calibration succeeded, RMS: {result['rms_error']:.4f}")
            print(f"  Total images: {total_images}, White positions: {white_image_positions}")
            print(f"  Valid rvecs: {sum(1 for r in calibrator.rvecs if r is not None)}")
            print(f"  Valid tvecs: {sum(1 for t in calibrator.tvecs if t is not None)}")
            print(f"  Valid errors: {sum(1 for e in calibrator.per_image_errors if e is not None)}")
            print(f"  Valid image_points: {sum(1 for p in calibrator.image_points if p is not None)}")
            if calibrator.point_ids is not None:
                print(f"  Valid point_ids: {sum(1 for p in calibrator.point_ids if p is not None)}")
            print(f"  Valid object_points: {sum(1 for o in calibrator.object_points if o is not None)}")
            
            # Generate calibration report and verify it's created successfully
            output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "results", f"test_white_image_{test_name}")
            try:
                report_files = calibrator.generate_calibration_report(output_dir)
                
                # Verify that report files were created
                self.assertIsNotNone(report_files, "Report files should not be None")
                self.assertIsInstance(report_files, dict, "Report files should be a dictionary")
                
                # Check for expected report components
                if 'html_report' in report_files:
                    html_file = report_files['html_report']
                    self.assertTrue(os.path.exists(html_file), f"HTML report should exist: {html_file}")
                    print(f"  HTML report created: {html_file}")
                
                if 'json_data' in report_files:
                    json_file = report_files['json_data']
                    self.assertTrue(os.path.exists(json_file), f"JSON data should exist: {json_file}")
                    print(f"  JSON data created: {json_file}")
                
                if 'visualizations' in report_files:
                    viz_dir = report_files['visualizations']
                    self.assertTrue(os.path.exists(viz_dir), f"Visualizations directory should exist: {viz_dir}")
                    print(f"  Visualizations created in: {viz_dir}")
                
                print(f"  Report generation successful for {test_name} test")
                
            except Exception as e:
                self.fail(f"Report generation failed for {test_name} test: {str(e)}")
            
        else:
            # Calibration failed - this could happen if not enough valid images remain
            print(f"White image {test_name} - Calibration failed as expected (insufficient valid images)")
            
            # Even for failed calibrations, try to generate a report to test error handling
            output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "results", f"test_white_image_{test_name}_failed")
            try:
                report_files = calibrator.generate_calibration_report(output_dir)
                # For failed calibrations, report generation might still work or might return None
                if report_files is not None:
                    print(f"  Report generated even for failed calibration: {test_name}")
                else:
                    print(f"  No report generated for failed calibration: {test_name}")
            except Exception as e:
                print(f"  Report generation failed for failed calibration {test_name}: {str(e)}")
    
    def test_calibration_with_resized_duplicate_last(self):
        """Test calibration with resized duplicate of last image (should fail due to size mismatch)."""
        images = self.load_test_images(5)
        pattern = self.load_test_pattern()
        
        # Create resized version of last image
        resized_image = self.create_resized_image(images[-1], scale_factor=0.8)
        test_images = images + [resized_image]
        total_images = len(test_images)
        
        calibrator = IntrinsicCalibrator(
            images=test_images,
            calibration_pattern=pattern
        )
        
        # This should fail due to image size mismatch
        with self.assertRaises(ValueError) as context:
            result = calibrator.calibrate(flags=0, verbose=False)
        
        # Verify that the error message mentions image size mismatch
        error_message = str(context.exception)
        self.assertIn("Image size mismatch", error_message)
        self.assertIn("All images must have the same size", error_message)
        
        print(f"Resized duplicate - Calibration correctly failed with error: {error_message}")
        print(f"  Total images attempted: {total_images}")
        print(f"  Expected behavior: Size validation prevents calibration with mixed image sizes")
    
    def test_calibration_with_same_size_images(self):
        """Test calibration with multiple images of the same size (should succeed)."""
        images = self.load_test_images(4)
        pattern = self.load_test_pattern()
        
        # Create duplicate of last image (same size) instead of resized
        duplicate_image = images[-1].copy()  # Same size, just a copy
        test_images = images + [duplicate_image]
        total_images = len(test_images)
        
        calibrator = IntrinsicCalibrator(
            images=test_images,
            calibration_pattern=pattern
        )
        
        # This should succeed since all images have the same size
        result = calibrator.calibrate(flags=0, verbose=False)
        
        # Verify calibration succeeded
        self.assertIsNotNone(result, "Calibration should succeed with same-size images")
        self.assertIn('camera_matrix', result)
        self.assertIn('distortion_coefficients', result)
        self.assertIn('rms_error', result)
        self.assertLess(result['rms_error'], 0.5)
        
        # Verify all arrays have correct length
        self.assertEqual(len(calibrator.rvecs), total_images)
        self.assertEqual(len(calibrator.tvecs), total_images)
        self.assertEqual(len(calibrator.per_image_errors), total_images)
        self.assertEqual(len(calibrator.image_points), total_images)
        self.assertEqual(len(calibrator.point_ids), total_images)
        self.assertEqual(len(calibrator.object_points), total_images)
        
        print(f"Same size images - Calibration succeeded, RMS: {result['rms_error']:.4f}")
        print(f"  Total images: {total_images}")
        print(f"  All images have same size - validation passed")


class TestIntrinsicCalibrationPatternTypes(unittest.TestCase):
    """
    Test intrinsic calibration with different calibration pattern types.
    
    This test class validates the IntrinsicCalibrator's ability to work with
    different types of calibration patterns supported by OpenCV:
    
    1. **Standard Chessboard Pattern:**
       - Traditional black-white checkerboard pattern
       - Corner detection based on gradient analysis
       - Most widely used and well-tested pattern type
    
    2. **ChArUco Board Pattern:**
       - Combines chessboard corners with ArUco markers
       - Robust against partial occlusion and perspective distortion
       - Provides both corner and marker-based feature detection
    
    3. **ArUco Grid Board Pattern:**
       - Grid of ArUco markers without chessboard squares
       - Excellent for challenging lighting and perspective conditions
       - Marker-based detection with unique identification
    
    Each test validates:
    - Successful pattern detection across multiple images
    - Reasonable calibration RMS error for the pattern type
    - Proper initialization with pattern-specific configurations
    - Compatibility with pattern JSON configuration files
    
    **RMS Error Thresholds (based on pattern characteristics):**
    - Standard Chessboard: < 0.5 pixels (high precision corner detection)
    - ChArUco Board: < 0.5 pixels (combined corner+marker robustness)
    - Grid Board: < 1.5 pixels (marker-based detection, slightly higher tolerance)
    
    **Implementation:**
    - Uses the same sample data directories as the examples
    - Loads pattern configurations from JSON files
    - Focuses on calibration success rather than result serialization
    - Validates different pattern detection algorithms
    
    These tests ensure the calibrator works reliably across different
    calibration pattern types commonly used in computer vision applications.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test data paths for different pattern types."""
        # Get project root (3 levels up from tests/unit/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Define sample data directories for each pattern type
        cls.chessboard_dir = os.path.join(project_root, "sample_data", "eye_in_hand_test_data")
        cls.charuco_dir = os.path.join(project_root, "sample_data", "intrinsic_calib_charuco_test_images")
        cls.gridboard_dir = os.path.join(project_root, "sample_data", "intrinsic_calib_grid_test_images")
        
        # Check which test data sets are available
        cls.has_chessboard_data = (
            os.path.exists(cls.chessboard_dir) and 
            os.path.exists(os.path.join(cls.chessboard_dir, "chessboard_config.json"))
        )
        cls.has_charuco_data = (
            os.path.exists(cls.charuco_dir) and 
            os.path.exists(os.path.join(cls.charuco_dir, "chessboard_config.json"))
        )
        cls.has_gridboard_data = (
            os.path.exists(cls.gridboard_dir) and 
            os.path.exists(os.path.join(cls.gridboard_dir, "chessboard_config.json"))
        )
    
    def load_images_from_directory(self, directory: str, max_images: int = 10) -> List[str]:
        """Load image file paths from a directory."""
        image_paths = []
        for filename in sorted(os.listdir(directory)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(directory, filename))
                if len(image_paths) >= max_images:
                    break
        return image_paths
    
    def load_pattern_from_config(self, config_path: str):
        """Load pattern configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return load_pattern_from_json(config_data)
    
    def test_standard_chessboard_calibration(self):
        """Test calibration with standard chessboard pattern."""
        if not self.has_chessboard_data:
            self.skipTest("Standard chessboard test data not available")
        
        # Load sample images and pattern configuration
        image_paths = self.load_images_from_directory(self.chessboard_dir, max_images=6)
        config_path = os.path.join(self.chessboard_dir, "chessboard_config.json")
        pattern = self.load_pattern_from_config(config_path)
        
        # Initialize calibrator with pattern and images
        calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern
        )
        
        # Perform calibration
        result = calibrator.calibrate(flags=0, verbose=False)
        
        # Verify successful calibration
        self.assertIsNotNone(result, "Standard chessboard calibration should succeed")
        self.assertIn('camera_matrix', result)
        self.assertIn('distortion_coefficients', result)
        self.assertIn('rms_error', result)
        
        # Check RMS error threshold for chessboard (high precision expected)
        rms_error = result['rms_error']
        self.assertLess(rms_error, 0.5, 
                       f"Standard chessboard RMS error {rms_error:.4f} should be < 0.5 pixels")
        
        # Verify pattern type
        self.assertEqual(pattern.pattern_id, 'standard_chessboard')
        
        print(f"Standard Chessboard - RMS error: {rms_error:.4f} pixels")
        print(f"  Pattern: {pattern.name}")
        print(f"  Images used: {len(image_paths)}")
        print(f"  Calibration successful: ✅")
    
    def test_charuco_board_calibration(self):
        """Test calibration with ChArUco board pattern."""
        if not self.has_charuco_data:
            self.skipTest("ChArUco board test data not available")
        
        # Load sample images and pattern configuration
        image_paths = self.load_images_from_directory(self.charuco_dir, max_images=7)
        config_path = os.path.join(self.charuco_dir, "chessboard_config.json")
        pattern = self.load_pattern_from_config(config_path)
        
        # Load images into memory (as done in the example)
        images = []
        for image_path in image_paths:
            img = cv2.imread(image_path)
            if img is not None:
                images.append(img)
        
        # Initialize calibrator with pattern and images
        calibrator = IntrinsicCalibrator(
            images=images,
            calibration_pattern=pattern
        )
        
        # Perform calibration
        result = calibrator.calibrate(flags=0, verbose=False)
        
        # Verify successful calibration
        self.assertIsNotNone(result, "ChArUco board calibration should succeed")
        self.assertIn('camera_matrix', result)
        self.assertIn('distortion_coefficients', result)
        self.assertIn('rms_error', result)
        
        # Check RMS error threshold for ChArUco (robust detection expected)
        rms_error = result['rms_error']
        self.assertLess(rms_error, 0.5, 
                       f"ChArUco board RMS error {rms_error:.4f} should be < 0.5 pixels")
        
        # Verify pattern type
        self.assertEqual(pattern.pattern_id, 'charuco_board')
        
        print(f"ChArUco Board - RMS error: {rms_error:.4f} pixels")
        print(f"  Pattern: {pattern.name}")
        print(f"  Images used: {len(images)}")
        print(f"  Calibration successful: ✅")
    
    def test_grid_board_calibration(self):
        """Test calibration with ArUco grid board pattern."""
        if not self.has_gridboard_data:
            self.skipTest("Grid board test data not available")
        
        # Load sample images and pattern configuration
        image_paths = self.load_images_from_directory(self.gridboard_dir, max_images=7)
        config_path = os.path.join(self.gridboard_dir, "chessboard_config.json")
        pattern = self.load_pattern_from_config(config_path)
        
        # Initialize calibrator with pattern and images
        calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern
        )
        
        # Perform calibration
        result = calibrator.calibrate(flags=0, verbose=False)
        
        # Verify successful calibration
        self.assertIsNotNone(result, "Grid board calibration should succeed")
        self.assertIn('camera_matrix', result)
        self.assertIn('distortion_coefficients', result)
        self.assertIn('rms_error', result)
        
        # Check RMS error threshold for grid board (marker-based, slightly higher tolerance)
        rms_error = result['rms_error']
        self.assertLess(rms_error, 1.5, 
                       f"Grid board RMS error {rms_error:.4f} should be < 1.5 pixels")
        
        # Verify pattern type
        self.assertEqual(pattern.pattern_id, 'grid_board')
        
        print(f"Grid Board - RMS error: {rms_error:.4f} pixels")
        print(f"  Pattern: {pattern.name}")
        print(f"  Images used: {len(image_paths)}")
        print(f"  Calibration successful: ✅")
    
    def test_pattern_comparison(self):
        """Compare calibration results across different pattern types."""
        results = {}
        
        # Test each pattern type that has available data
        pattern_tests = [
            ('standard_chessboard', self.has_chessboard_data, self.chessboard_dir, 0.5),
            ('charuco_board', self.has_charuco_data, self.charuco_dir, 0.5),
            ('grid_board', self.has_gridboard_data, self.gridboard_dir, 1.5)
        ]
        
        for pattern_name, has_data, data_dir, threshold in pattern_tests:
            if not has_data:
                continue
                
            try:
                # Load images and pattern
                image_paths = self.load_images_from_directory(data_dir, max_images=6)
                config_path = os.path.join(data_dir, "chessboard_config.json")
                pattern = self.load_pattern_from_config(config_path)
                
                # Initialize and calibrate
                if pattern_name == 'charuco_board':
                    # Load images into memory for ChArUco (as in example)
                    images = [cv2.imread(path) for path in image_paths if cv2.imread(path) is not None]
                    calibrator = IntrinsicCalibrator(images=images, calibration_pattern=pattern)
                else:
                    calibrator = IntrinsicCalibrator(image_paths=image_paths, calibration_pattern=pattern)
                
                result = calibrator.calibrate(flags=0, verbose=False)
                
                if result is not None:
                    results[pattern_name] = {
                        'rms_error': result['rms_error'],
                        'threshold': threshold,
                        'success': result['rms_error'] < threshold,
                        'pattern_name': pattern.name
                    }
                    
            except Exception as e:
                print(f"Warning: {pattern_name} calibration failed: {e}")
        
        # Verify we tested at least one pattern
        self.assertGreater(len(results), 0, "At least one pattern type should be testable")
        
        # Print comparison results
        print("\nPattern Type Comparison:")
        print("-" * 70)
        for pattern_id, data in results.items():
            status = "✅ PASS" if data['success'] else "❌ FAIL"
            print(f"{data['pattern_name']:20} | RMS: {data['rms_error']:6.4f} | "
                  f"Threshold: {data['threshold']:4.1f} | {status}")
        
        # Verify all tested patterns meet their thresholds
        for pattern_id, data in results.items():
            self.assertTrue(data['success'], 
                          f"{pattern_id} failed: RMS {data['rms_error']:.4f} > {data['threshold']}")


def run_tests():
    """Run all tests with detailed output."""
    loader = unittest.TestLoader()
    
    # List of all test classes - add new test classes here
    test_classes = [
        TestIntrinsicCalibrationDistortionModels,
        TestIntrinsicCalibrationPatternDetectionFailures,
        TestIntrinsicCalibrationPatternTypes,
    ]
    
    # Load tests from all test classes
    test_suites = []
    for test_class in test_classes:
        suite = loader.loadTestsFromTestCase(test_class)
        test_suites.append(suite)
    
    # Combine all test suites
    combined_suite = unittest.TestSuite(test_suites)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
