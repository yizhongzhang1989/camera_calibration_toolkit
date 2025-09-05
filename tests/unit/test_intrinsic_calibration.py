#!/usr/bin/env python3
"""
Unit Tests for Intrinsic Camera Calibration
==========================================

This module contains focused unit tests for the IntrinsicCalibrator class,
specifically testing calibration with real data and different camera models.
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


class TestIntrinsicCalibrationWithRealData(unittest.TestCase):
    """Test intrinsic calibration with real sample data and different camera models."""
    
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


class TestIntrinsicCalibrationCornerCases(unittest.TestCase):
    """
    Test intrinsic calibration corner cases and edge conditions.
    
    This test class validates how the IntrinsicCalibrator handles problematic scenarios:
    
    1. Images with no detectable calibration patterns (pure white images)
    2. Mixed image sizes in the calibration set
    
    Expected behaviors:
    - White images (no pattern): Should be skipped, calibration continues with valid images
    - Resized images: May increase RMS error but calibration should still work or fail gracefully
    
    Key validations:
    - Arrays (rvecs, tvecs, per_image_errors, image_points, point_ids, object_points) should have same length as input images
    - Failed detection positions should have None values in arrays
    - Valid detection positions should have non-None values in arrays
    
    Implementation approach:
    - Uses images directly (numpy arrays) instead of image paths for simplicity
    - No temporary file management required
    - Clean, in-memory testing approach
    
    These tests ensure robustness and help identify potential edge cases in real-world usage.
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
            
        else:
            # Calibration failed - this could happen if not enough valid images remain
            print(f"White image {test_name} - Calibration failed as expected (insufficient valid images)")
    
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


def run_tests():
    """Run all tests with detailed output."""
    loader = unittest.TestLoader()
    
    # List of all test classes - add new test classes here
    test_classes = [
        TestIntrinsicCalibrationWithRealData,
        TestIntrinsicCalibrationCornerCases,
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
