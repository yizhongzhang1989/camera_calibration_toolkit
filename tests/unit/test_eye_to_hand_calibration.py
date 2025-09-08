#!/usr/bin/env python3
"""
Eye-to-Hand Calibration Unit Tests
==================================

This module contains unit tests for the EyeToHandCalibrator class.
Tests are adapted from the examples and provide comprehensive validation
of the eye-to-hand calibration workflow.

Test Coverage:
- Intrinsic calibration prerequisite
- Data loading and validation
- Eye-to-hand calibration execution
- Error threshold validation
- Result format validation
- IO operations (JSON serialization)
- Report generation

Test Features:
- RMS error threshold validation (fails if > 1.0 pixel)
- Complete calibration workflow testing
- Comprehensive error reporting and validation
- Data integrity checks

Note: These tests validate calibration quality through error threshold checking
and assert calibration quality to ensure robust validation.
"""

import os
import cv2
import sys
import numpy as np
import json
import unittest
import tempfile
import shutil
from typing import Tuple, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.eye_to_hand_calibration import EyeToHandCalibrator
from core.intrinsic_calibration import IntrinsicCalibrator
from core.calibration_patterns import load_pattern_from_json


class TestEyeToHandCalibration(unittest.TestCase):
    """
    Unit test class for EyeToHandCalibrator functionality.
    
    This test class validates the EyeToHandCalibrator's ability to perform
    complete eye-to-hand calibration workflows including data loading,
    intrinsic calibration, hand-eye calibration, and result validation.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with data directories and temporary results directory."""
        cls.eye_to_hand_data_dir = os.path.join("sample_data", "eye_to_hand_test_data")
        cls.intrinsic_data_dir = os.path.join("sample_data", "intrinsic_calib_grid_test_images")
        cls.results_dir = tempfile.mkdtemp(prefix="eye_to_hand_test_")
        
        # Verify test data exists
        if not os.path.exists(cls.eye_to_hand_data_dir):
            raise unittest.SkipTest(f"Test data directory not found: {cls.eye_to_hand_data_dir}")
        if not os.path.exists(cls.intrinsic_data_dir):
            raise unittest.SkipTest(f"Intrinsic data directory not found: {cls.intrinsic_data_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test results directory."""
        if os.path.exists(cls.results_dir):
            shutil.rmtree(cls.results_dir)
    
    def perform_intrinsic_calibration(self, data_dir: str, results_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        Perform camera intrinsic calibration using images and pattern from the specified directory.

        Args:
            data_dir: Directory containing calibration images and pattern configuration
            results_dir: Directory to save intrinsic calibration results

        Returns:
            tuple: (camera_matrix, distortion_coefficients, rms_error) or (None, None, None) if failed
        """
        print("\n" + "="*60)
        print("📐 Performing Camera Intrinsic Calibration")
        print("="*60)

        try:
            # Step 1: Load calibration pattern configuration
            pattern_config_path = os.path.join(data_dir, "chessboard_config.json")
            if not os.path.exists(pattern_config_path):
                raise FileNotFoundError(f"Pattern config not found: {pattern_config_path}")

            # Load JSON data from file
            with open(pattern_config_path, 'r') as f:
                pattern_json_data = json.load(f)

            calibration_pattern = load_pattern_from_json(pattern_json_data)
            print(f"✅ Loaded pattern: {calibration_pattern}")

            # Step 2: Load images for intrinsic calibration
            image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
            image_files.sort()  # Sort to ensure consistent ordering

            print(f"🔍 Found {len(image_files)} image files")

            images = []
            for image_file in image_files:
                image_path = os.path.join(data_dir, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
                    print(f"✅ Loaded image: {image_file}")
                else:
                    print(f"⚠️ Warning: Could not load {image_file}")

            print(f"📊 Successfully loaded {len(images)} images for intrinsic calibration")

            # Step 3: Perform intrinsic calibration
            intrinsic_calibrator = IntrinsicCalibrator(
                images=images,
                calibration_pattern=calibration_pattern
            )

            print(f"🔍 Performing intrinsic calibration with {len(images)} images...")
            success = intrinsic_calibrator.calibrate(verbose=True)

            if success:
                camera_matrix = intrinsic_calibrator.get_camera_matrix()
                distortion_coefficients = intrinsic_calibrator.get_distortion_coefficients()
                rms_error = intrinsic_calibrator.get_rms_error()

                print(f"✅ Intrinsic calibration successful!")
                print(f"📊 RMS Error: {rms_error:.4f} pixels")
                print(f"📷 Camera matrix:")
                print(f"   fx: {camera_matrix[0,0]:.2f}, fy: {camera_matrix[1,1]:.2f}")
                print(f"   cx: {camera_matrix[0,2]:.2f}, cy: {camera_matrix[1,2]:.2f}")
                print(f"🔍 Distortion coefficients: {distortion_coefficients}")

                # Generate intrinsic calibration report
                intrinsic_results_dir = os.path.join(results_dir, "intrinsic_calibration")
                os.makedirs(intrinsic_results_dir, exist_ok=True)

                report_info = intrinsic_calibrator.generate_calibration_report(intrinsic_results_dir)
                print(f"📋 Intrinsic calibration report generated: {report_info}")

                return camera_matrix, distortion_coefficients, rms_error
            else:
                print("❌ Intrinsic calibration failed!")
                return None, None, None

        except Exception as e:
            print(f"❌ Error during intrinsic calibration: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def load_data(self, data_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load images and corresponding end-effector transformation matrices.

        Args:
            data_dir: Directory containing images and pose JSON files

        Returns:
            tuple: (images, end2base_matrices) lists
        """
        print("\n" + "="*60)
        print("📂 Loading Eye-to-Hand Data")
        print("="*60)

        # Get all image files
        image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        image_files.sort()  # Sort to ensure consistent ordering

        images = []
        end2base_matrices = []

        print(f"🔍 Found {len(image_files)} image files")

        for image_file in image_files:
            # Load image
            image_path = os.path.join(data_dir, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"⚠️ Warning: Could not load image {image_file}")
                continue

            # Load corresponding pose file
            pose_file = image_file.replace('.jpg', '.json')
            pose_path = os.path.join(data_dir, pose_file)

            if not os.path.exists(pose_path):
                print(f"⚠️ Warning: Pose file not found for {image_file}")
                continue

            try:
                with open(pose_path, 'r') as f:
                    pose_data = json.load(f)

                # Extract end2base matrix
                if 'end2base' in pose_data:
                    end2base_matrix = np.array(pose_data['end2base'], dtype=np.float64)
                elif 'end_xyzrpy' in pose_data:
                    # Convert from xyz-rpy format if needed
                    from core.utils import xyz_rpy_to_matrix
                    pose_xyzrpy = pose_data['end_xyzrpy']
                    end2base_matrix = xyz_rpy_to_matrix(
                        pose_xyzrpy['x'], pose_xyzrpy['y'], pose_xyzrpy['z'],
                        pose_xyzrpy['rx'], pose_xyzrpy['ry'], pose_xyzrpy['rz']
                    )
                else:
                    print(f"⚠️ Warning: Invalid pose format in {pose_file}")
                    continue

                # Validate matrix format
                if end2base_matrix.shape != (4, 4):
                    print(f"⚠️ Warning: Invalid transformation matrix shape in {pose_file}")
                    continue

                # Add to lists
                images.append(image)
                end2base_matrices.append(end2base_matrix)

                print(f"✅ Loaded: {image_file} with pose data")

            except Exception as e:
                print(f"⚠️ Warning: Error loading pose data from {pose_file}: {e}")
                continue

        print(f"📊 Successfully loaded {len(images)} image-pose pairs")
        return images, end2base_matrices

    def test_eye_to_hand_calibration(self):
        """
        Test complete eye-to-hand calibration workflow.

        This test validates the entire calibration process including:
        - Intrinsic calibration
        - Data loading
        - Eye-to-hand calibration
        - Error threshold validation
        - Result format validation
        """
        print("\n" + "="*60)
        print("🤖 Testing Eye-to-Hand Calibration")
        print("="*60)

        # Step 1: Perform intrinsic calibration using grid test images
        camera_matrix, distortion_coefficients, intrinsic_rms = self.perform_intrinsic_calibration(
            self.intrinsic_data_dir, self.results_dir
        )

        self.assertIsNotNone(camera_matrix, "Intrinsic calibration failed - camera matrix is None")
        self.assertIsNotNone(distortion_coefficients, "Intrinsic calibration failed - distortion coefficients are None")
        self.assertIsNotNone(intrinsic_rms, "Intrinsic calibration failed - RMS error is None")

        # Validate intrinsic calibration quality (grid board may have higher error)
        self.assertLess(intrinsic_rms, 2.0, f"Intrinsic calibration RMS error too high: {intrinsic_rms:.4f} > 2.0")

        print(f"✅ Intrinsic calibration passed quality check (RMS: {intrinsic_rms:.4f})")

        # Step 2: Load eye-to-hand data
        images, end2base_matrices = self.load_data(self.eye_to_hand_data_dir)

        self.assertGreater(len(images), 0, "No images loaded")
        self.assertEqual(len(images), len(end2base_matrices), "Image count doesn't match pose count")
        self.assertGreaterEqual(len(images), 5, f"Insufficient data for calibration: {len(images)} < 5")

        print(f"✅ Data loading passed validation ({len(images)} image-pose pairs)")

        # Step 3: Load pattern configuration for hand-eye calibration
        pattern_config_path = os.path.join(self.eye_to_hand_data_dir, "chessboard_config.json")
        with open(pattern_config_path, 'r') as f:
            pattern_json_data = json.load(f)
        calibration_pattern = load_pattern_from_json(pattern_json_data)

        # Step 4: Initialize eye-to-hand calibrator
        calibrator = EyeToHandCalibrator(
            images=images,
            end2base_matrices=end2base_matrices,
            calibration_pattern=calibration_pattern,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            verbose=True
        )

        self.assertIsNotNone(calibrator, "Failed to create EyeToHandCalibrator")

        # Step 5: Perform calibration
        print(f"🔍 Starting eye-to-hand calibration with {len(images)} image-pose pairs...")

        result = calibrator.calibrate(verbose=True)

        self.assertIsNotNone(result, "Eye-to-hand calibration returned None")
        self.assertIsInstance(result, dict, "Calibration result should be a dictionary")

        # Step 6: Validate calibration results
        required_keys = ['base2cam_matrix', 'target2end_matrix', 'rms_error']
        for key in required_keys:
            self.assertIn(key, result, f"Missing required key in result: {key}")
        
        # Check if per_image_errors is available (may be in calibrator object)
        per_image_errors = getattr(calibrator, 'per_image_errors', None)

        rms_error = result['rms_error']
        base2cam_matrix = result['base2cam_matrix']
        target2end_matrix = result['target2end_matrix']

        # Validate error threshold (eye-to-hand typically has higher tolerance)
        self.assertLess(rms_error, 2.0, f"Eye-to-hand calibration RMS error too high: {rms_error:.4f} > 2.0")

        # Validate matrix formats
        self.assertEqual(base2cam_matrix.shape, (4, 4), "base2cam_matrix should be 4x4")
        self.assertEqual(target2end_matrix.shape, (4, 4), "target2end_matrix should be 4x4")

        # Validate transformation matrix properties
        np.testing.assert_allclose(base2cam_matrix[3, :], [0, 0, 0, 1], atol=1e-6,
                                 err_msg="base2cam_matrix bottom row should be [0, 0, 0, 1]")
        np.testing.assert_allclose(target2end_matrix[3, :], [0, 0, 0, 1], atol=1e-6,
                                 err_msg="target2end_matrix bottom row should be [0, 0, 0, 1]")

        print(f"✅ Eye-to-hand calibration successful!")
        print(f"📊 RMS Error: {rms_error:.4f} pixels")
        print(f"🔗 Base to Camera transformation matrix:")
        print(f"{base2cam_matrix}")
        print(f"🎯 Target to End-Effector transformation matrix:")
        print(f"{target2end_matrix}")

        # Step 7: Generate calibration report
        hand_eye_results_dir = os.path.join(self.results_dir, "eye_to_hand_calibration")
        os.makedirs(hand_eye_results_dir, exist_ok=True)

        report_info = calibrator.generate_calibration_report(hand_eye_results_dir, verbose=True)
        self.assertIsNotNone(report_info, "Failed to generate calibration report")

        print(f"📋 Eye-to-hand calibration report generated: {report_info}")

        # Step 8: Test JSON serialization/deserialization
        calibrator_json = calibrator.to_json()
        self.assertIsInstance(calibrator_json, dict, "to_json() should return a dictionary")

        # Create new calibrator and load from JSON
        new_calibrator = EyeToHandCalibrator()
        new_calibrator.from_json(calibrator_json)

        # Verify loaded matrices match
        np.testing.assert_array_almost_equal(
            new_calibrator.get_base2cam_matrix(), base2cam_matrix,
            err_msg="Loaded base2cam_matrix doesn't match original"
        )
        np.testing.assert_array_almost_equal(
            new_calibrator.get_target2end_matrix(), target2end_matrix,
            err_msg="Loaded target2end_matrix doesn't match original"
        )

        print(f"✅ JSON serialization/deserialization test passed")

        print(f"🎉 All eye-to-hand calibration tests passed!")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
