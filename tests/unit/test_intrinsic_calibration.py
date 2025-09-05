#!/usr/bin/env python3
"""
Unit Tests for Intrinsic Camera Calibration
==========================================

This module contains comprehensive unit tests for the IntrinsicCalibrator class.
Tests cover initialization, calibration with different patterns, error handling,
and result validation.

Test Structure:
- TestIntrinsicCalibratorInit: Constructor and initialization tests
- TestIntrinsicCalibratorCalibration: Core calibration functionality tests
- TestIntrinsicCalibratorPatterns: Different calibration pattern tests
- TestIntrinsicCalibratorErrorHandling: Error condition and edge case tests
- TestIntrinsicCalibratorResults: Result validation and getter method tests
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
import numpy as np
import cv2
from typing import List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import StandardChessboard, load_pattern_from_json


class TestIntrinsicCalibratorInit(unittest.TestCase):
    """Test IntrinsicCalibrator initialization and setup."""
    
    def test_init_empty(self):
        """Test initialization with no arguments."""
        calibrator = IntrinsicCalibrator()
        
        self.assertIsNone(calibrator.images)
        self.assertIsNone(calibrator.image_paths)
        self.assertIsNone(calibrator.calibration_pattern)
        self.assertFalse(calibrator.is_calibrated())
    
    def test_init_with_pattern(self):
        """Test initialization with calibration pattern."""
        pattern = StandardChessboard(width=9, height=6, square_size=0.025)
        calibrator = IntrinsicCalibrator(calibration_pattern=pattern)
        
        self.assertIsNotNone(calibrator.calibration_pattern)
        self.assertEqual(calibrator.calibration_pattern.width, 9)
        self.assertEqual(calibrator.calibration_pattern.height, 6)
    
    def test_init_with_arrays(self):
        """Test initialization with image arrays."""
        # Create synthetic test images
        test_images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        calibrator = IntrinsicCalibrator(images=test_images)
        
        self.assertIsNotNone(calibrator.images)
        self.assertEqual(len(calibrator.images), 3)
        self.assertEqual(calibrator.image_size, (640, 480))
    
    def test_set_calibration_pattern(self):
        """Test setting calibration pattern after initialization."""
        calibrator = IntrinsicCalibrator()
        pattern = StandardChessboard(width=8, height=6, square_size=0.02)
        
        calibrator.set_calibration_pattern(pattern)
        
        self.assertIsNotNone(calibrator.calibration_pattern)
        self.assertEqual(calibrator.calibration_pattern.square_size, 0.02)


class TestIntrinsicCalibratorCalibration(unittest.TestCase):
    """Test core calibration functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data paths and patterns."""
        # Get project root (3 levels up from tests/unit/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cls.test_data_dir = os.path.join(project_root, "sample_data", "eye_in_hand_test_data")
        cls.pattern_config_path = os.path.join(cls.test_data_dir, "chessboard_config.json")
        cls.images_dir = cls.test_data_dir  # Images are directly in the test data dir
        
        # Check if test data exists
        cls.has_test_data = (
            os.path.exists(cls.pattern_config_path) and 
            os.path.exists(cls.images_dir)
        )
    
    def setUp(self):
        """Set up individual test."""
        if not self.has_test_data:
            self.skipTest("Test data not available")
    
    def load_test_images(self, max_images: int = 6) -> List[str]:
        """Load test image paths."""
        image_paths = []
        for filename in sorted(os.listdir(self.images_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(self.images_dir, filename))
                if len(image_paths) >= max_images:
                    break
        return image_paths
    
    def load_test_pattern(self):
        """Load test calibration pattern."""
        with open(self.pattern_config_path, 'r') as f:
            pattern_json = json.load(f)
        return load_pattern_from_json(pattern_json)
    
    def test_calibrate_basic(self):
        """Test basic calibration functionality."""
        image_paths = self.load_test_images(6)
        pattern = self.load_test_pattern()
        
        self.assertGreaterEqual(len(image_paths), 3)
        
        calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern
        )
        
        result = calibrator.calibrate(verbose=False)
        
        # Test successful calibration
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        
        # Test return dictionary structure
        required_keys = {'camera_matrix', 'distortion_coefficients', 'rms_error'}
        self.assertEqual(set(result.keys()), required_keys)
        
        # Test result types
        self.assertIsInstance(result['camera_matrix'], np.ndarray)
        self.assertIsInstance(result['distortion_coefficients'], np.ndarray)
        self.assertIsInstance(result['rms_error'], float)
        
        # Test matrix shapes
        self.assertEqual(result['camera_matrix'].shape, (3, 3))
        
        # Distortion coefficients can be 1D or 2D array - check appropriately
        dist_coeffs = result['distortion_coefficients']
        if len(dist_coeffs.shape) == 2:
            # 2D array (Nx1 or 1xN)
            total_coeffs = max(dist_coeffs.shape)
        else:
            # 1D array
            total_coeffs = len(dist_coeffs)
        
        self.assertGreaterEqual(total_coeffs, 4)  # At least 4 distortion coefficients
        
        # Test calibration quality
        self.assertGreater(result['rms_error'], 0)
        self.assertLess(result['rms_error'], 5.0)  # Reasonable RMS error threshold
        
        # Test class state
        self.assertTrue(calibrator.is_calibrated())
        self.assertIsNotNone(calibrator.camera_matrix)
        self.assertIsNotNone(calibrator.distortion_coefficients)
        self.assertIsNotNone(calibrator.rms_error)
    
    def test_calibrate_with_flags(self):
        """Test calibration with different flags."""
        image_paths = self.load_test_images(6)
        pattern = self.load_test_pattern()
        
        calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern
        )
        
        # Test with different distortion models
        result_standard = calibrator.calibrate(flags=0, verbose=False)
        self.assertIsNotNone(result_standard)
        
        # Reset calibrator for next test
        calibrator.calibration_completed = False
        calibrator.camera_matrix = None
        calibrator.distortion_coefficients = None
        
        # Test with rational model
        result_rational = calibrator.calibrate(
            flags=cv2.CALIB_RATIONAL_MODEL, 
            verbose=False
        )
        self.assertIsNotNone(result_rational)
        
        # Compare results
        self.assertEqual(result_standard['camera_matrix'].shape, (3, 3))
        self.assertEqual(result_rational['camera_matrix'].shape, (3, 3))
        
        # Rational model should have more distortion coefficients
        self.assertGreaterEqual(
            result_rational['distortion_coefficients'].shape[0],
            result_standard['distortion_coefficients'].shape[0]
        )
    
    def test_calibrate_with_initial_guess(self):
        """Test calibration with initial camera matrix guess."""
        image_paths = self.load_test_images(6)
        pattern = self.load_test_pattern()
        
        calibrator = IntrinsicCalibrator(
            image_paths=image_paths,
            calibration_pattern=pattern
        )
        
        # Create initial guess
        initial_matrix = np.array([
            [640, 0, 320],
            [0, 640, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        result = calibrator.calibrate(
            cameraMatrix=initial_matrix,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
            verbose=False
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result['camera_matrix'], np.ndarray)
        
        # Result should be different from initial guess (refined)
        self.assertFalse(np.array_equal(result['camera_matrix'], initial_matrix))


class TestIntrinsicCalibratorPatterns(unittest.TestCase):
    """Test calibration with different patterns."""
    
    def test_calibrate_chessboard_pattern(self):
        """Test calibration with standard chessboard pattern."""
        pattern = StandardChessboard(width=9, height=6, square_size=0.025)
        
        # Create synthetic images (will fail pattern detection, but tests API)
        test_images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        calibrator = IntrinsicCalibrator(images=test_images, calibration_pattern=pattern)
        
        # This should fail pattern detection but not crash
        with self.assertRaises(ValueError):
            calibrator.calibrate(verbose=False)
    
    def test_pattern_detection_automatic(self):
        """Test automatic pattern detection when not done explicitly."""
        pattern = StandardChessboard(width=9, height=6, square_size=0.025)
        test_images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        
        calibrator = IntrinsicCalibrator(images=test_images, calibration_pattern=pattern)
        
        # Pattern detection should be called automatically during calibration
        self.assertIsNone(calibrator.image_points)
        
        with self.assertRaises(ValueError):  # Will fail on synthetic images
            calibrator.calibrate(verbose=False)


class TestIntrinsicCalibratorErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_calibrate_no_images(self):
        """Test calibration with no images loaded."""
        pattern = StandardChessboard(width=9, height=6, square_size=0.025)
        calibrator = IntrinsicCalibrator(calibration_pattern=pattern)
        
        with self.assertRaises(ValueError):
            calibrator.calibrate(verbose=False)
    
    def test_calibrate_no_pattern(self):
        """Test calibration with no pattern set."""
        test_images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        calibrator = IntrinsicCalibrator(images=test_images)
        
        with self.assertRaises(ValueError):
            calibrator.calibrate(verbose=False)
    
    def test_calibrate_insufficient_images(self):
        """Test calibration with insufficient valid images."""
        pattern = StandardChessboard(width=9, height=6, square_size=0.025)
        # Only 2 images (less than minimum of 3)
        test_images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
        
        calibrator = IntrinsicCalibrator(images=test_images, calibration_pattern=pattern)
        
        with self.assertRaises(ValueError):
            calibrator.calibrate(verbose=False)
    
    def test_calibrate_invalid_image_paths(self):
        """Test calibration with invalid image paths."""
        pattern = StandardChessboard(width=9, height=6, square_size=0.025)
        invalid_paths = ["/nonexistent/path1.jpg", "/nonexistent/path2.jpg"]
        
        # This should fail during image loading
        result = IntrinsicCalibrator().set_images_from_paths(invalid_paths)
        self.assertFalse(result)


class TestIntrinsicCalibratorResults(unittest.TestCase):
    """Test result validation and getter methods."""
    
    def setUp(self):
        """Set up calibrator with mock results."""
        self.calibrator = IntrinsicCalibrator()
        
        # Mock calibration results
        self.calibrator.camera_matrix = np.array([
            [640, 0, 320],
            [0, 640, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.calibrator.distortion_coefficients = np.array(
            [-0.1, 0.05, 0.001, 0.002, -0.01], dtype=np.float32
        )
        
        self.calibrator.rms_error = 0.5
        self.calibrator.calibration_completed = True
        self.calibrator.per_image_errors = [0.4, 0.6, 0.5]
    
    def test_getter_methods(self):
        """Test getter methods for calibration results."""
        # Test RMS error getter
        self.assertEqual(self.calibrator.get_rms_error(), 0.5)
        
        # Test camera matrix getter
        camera_matrix = self.calibrator.get_camera_matrix()
        self.assertIsNotNone(camera_matrix)
        self.assertEqual(camera_matrix.shape, (3, 3))
        
        # Test distortion coefficients getter
        dist_coeffs = self.calibrator.get_distortion_coefficients()
        self.assertIsNotNone(dist_coeffs)
        self.assertEqual(len(dist_coeffs), 5)
        
        # Test per-image errors getter
        per_image_errors = self.calibrator.get_per_image_errors()
        self.assertEqual(len(per_image_errors), 3)
    
    def test_is_calibrated(self):
        """Test calibration status checking."""
        self.assertTrue(self.calibrator.is_calibrated())
        
        # Test with uncalibrated state
        uncalibrated = IntrinsicCalibrator()
        self.assertFalse(uncalibrated.is_calibrated())
    
    def test_save_and_load_calibration(self):
        """Test saving and loading calibration results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_calibration.json")
            
            # Save calibration
            self.calibrator.save_calibration(filepath)
            self.assertTrue(os.path.exists(filepath))
            
            # Load calibration into new calibrator
            new_calibrator = IntrinsicCalibrator()
            success = new_calibrator.load_calibration(filepath)
            
            self.assertTrue(success)
            self.assertTrue(new_calibrator.is_calibrated())
            
            # Compare results
            np.testing.assert_array_almost_equal(
                new_calibrator.camera_matrix,
                self.calibrator.camera_matrix
            )
            
            np.testing.assert_array_almost_equal(
                new_calibrator.distortion_coefficients,
                self.calibrator.distortion_coefficients
            )
            
            self.assertAlmostEqual(
                new_calibrator.rms_error,
                self.calibrator.rms_error
            )
    
    def test_save_results_directory(self):
        """Test save_results method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.calibrator.save_results(temp_dir)
            
            # Check if results file was created
            results_file = os.path.join(temp_dir, "intrinsic_calibration_results.json")
            self.assertTrue(os.path.exists(results_file))


class TestIntrinsicCalibratorSerialization(unittest.TestCase):
    """Test JSON serialization and deserialization."""
    
    def setUp(self):
        """Set up calibrator with complete mock data."""
        self.calibrator = IntrinsicCalibrator()
        
        # Set up complete mock calibration state
        self.calibrator.camera_matrix = np.eye(3, dtype=np.float32)
        self.calibrator.distortion_coefficients = np.zeros(5, dtype=np.float32)
        self.calibrator.rms_error = 0.5
        self.calibrator.calibration_completed = True
        self.calibrator.image_size = (640, 480)
        self.calibrator.distortion_model = 'standard'
        
        # Mock pattern
        self.calibrator.calibration_pattern = StandardChessboard(9, 6, 0.025)
    
    def test_to_json(self):
        """Test serialization to JSON."""
        json_data = self.calibrator.to_json()
        
        self.assertIsInstance(json_data, dict)
        self.assertIn('camera_matrix', json_data)
        self.assertIn('distortion_coefficients', json_data)
        self.assertIn('rms_error', json_data)
        self.assertIn('image_size', json_data)
        
        # Test that arrays are converted to lists
        self.assertIsInstance(json_data['camera_matrix'], list)
        self.assertIsInstance(json_data['distortion_coefficients'], list)
    
    def test_from_json(self):
        """Test deserialization from JSON."""
        # First serialize
        json_data = self.calibrator.to_json()
        
        # Create new calibrator and deserialize
        new_calibrator = IntrinsicCalibrator()
        new_calibrator.from_json(json_data)
        
        # Verify deserialization
        self.assertTrue(new_calibrator.is_calibrated())
        np.testing.assert_array_equal(
            new_calibrator.camera_matrix,
            self.calibrator.camera_matrix
        )
        np.testing.assert_array_equal(
            new_calibrator.distortion_coefficients,
            self.calibrator.distortion_coefficients
        )
        self.assertEqual(new_calibrator.rms_error, self.calibrator.rms_error)


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestIntrinsicCalibratorInit,
        TestIntrinsicCalibratorCalibration,
        TestIntrinsicCalibratorPatterns,
        TestIntrinsicCalibratorErrorHandling,
        TestIntrinsicCalibratorResults,
        TestIntrinsicCalibratorSerialization
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
