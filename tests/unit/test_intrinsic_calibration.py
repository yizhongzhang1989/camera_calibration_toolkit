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


def run_tests():
    """Run all tests with detailed output."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestIntrinsicCalibrationWithRealData)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
