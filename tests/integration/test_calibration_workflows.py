"""
Integration tests for calibration workflows.

Tests complete calibration workflows from end-to-end,
including file I/O, pattern detection, and calibration
with synthetic data to ensure components work together.
"""
import pytest
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil
from unittest.mock import patch, Mock
import cv2

from core.intrinsic_calibration import IntrinsicCalibrator
from core.eye_in_hand_calibration import EyeInHandCalibrator
from core.calibration_patterns import (
    create_standard_chessboard, 
    create_charuco_board, 
    create_aruco_grid_board,
    save_pattern_to_json,
    load_pattern_from_json
)

class TestIntrinsicCalibrationWorkflow:
    """Test complete intrinsic calibration workflows."""
    
    @pytest.mark.integration
    def test_complete_chessboard_calibration_workflow(self, temp_output_dir, synthetic_chessboard_images):
        """Test complete intrinsic calibration workflow with chessboard pattern."""
        # Create a chessboard pattern
        pattern = create_standard_chessboard(rows=7, columns=10, square_size=20.0)
        pattern_file = temp_output_dir / "chessboard_config.json"
        save_pattern_to_json(pattern, str(pattern_file))
        
        # Create calibrator and load pattern
        calibrator = IntrinsicCalibrator()
        loaded_pattern = load_pattern_from_json(str(pattern_file))
        
        # Set up calibration
        calibrator.calibration_pattern = loaded_pattern
        calibrator.pattern_type = 'standard_chessboard'
        
        # Mock the image detection process since we're using synthetic data
        with patch.object(calibrator, 'detect_pattern_in_images') as mock_detect:
            # Mock successful detection
            mock_object_points = [np.random.rand(70, 3) for _ in range(len(synthetic_chessboard_images))]
            mock_image_points = [np.random.rand(70, 2) for _ in range(len(synthetic_chessboard_images))]
            mock_detect.return_value = (mock_object_points, mock_image_points)
            
            # Mock OpenCV calibrateCamera
            with patch('cv2.calibrateCamera') as mock_calibrate:
                mock_calibrate.return_value = (
                    0.3,  # RMS error
                    np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float),  # Camera matrix
                    np.array([-0.1, 0.05, 0, 0, 0], dtype=float),  # Distortion coefficients
                    None,  # rvecs
                    None   # tvecs
                )
                
                # Set images and perform calibration
                calibrator.images = synthetic_chessboard_images
                calibrator.image_size = (640, 480)
                
                # Run the calibration
                rms_error = calibrator.calibrate_camera()
                
                # Verify results
                assert rms_error == 0.3
                assert calibrator.camera_matrix is not None
                assert calibrator.distortion_coefficients is not None
                assert calibrator.camera_matrix.shape == (3, 3)
                assert len(calibrator.distortion_coefficients) == 5
                
                # Test saving results
                results_file = temp_output_dir / "calibration_results.json"
                calibrator.save_calibration(str(results_file))
                assert results_file.exists()
    
    @pytest.mark.integration
    def test_complete_charuco_calibration_workflow(self, temp_output_dir):
        """Test complete intrinsic calibration workflow with ChArUco pattern."""
        # Create a ChArUco pattern
        pattern = create_charuco_board(
            squares_x=8, squares_y=6, 
            square_length=40.0, marker_length=30.0,
            dictionary_id=cv2.aruco.DICT_6X6_250
        )
        pattern_file = temp_output_dir / "charuco_config.json"
        save_pattern_to_json(pattern, str(pattern_file))
        
        # Create calibrator and load pattern
        calibrator = IntrinsicCalibrator()
        loaded_pattern = load_pattern_from_json(str(pattern_file))
        
        calibrator.calibration_pattern = loaded_pattern
        calibrator.pattern_type = 'charuco_board'
        
        # Create mock images
        mock_images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        
        # Mock the detection and calibration process
        with patch.object(calibrator, 'detect_pattern_in_images') as mock_detect, \
             patch('cv2.aruco.calibrateCameraCharuco') as mock_calibrate:
            
            # Mock successful ChArUco detection
            mock_all_corners = [np.random.rand(20, 2) for _ in range(5)]
            mock_all_ids = [np.random.randint(0, 48, (20, 1)) for _ in range(5)]
            mock_detect.return_value = (mock_all_corners, mock_all_ids)
            
            # Mock successful calibration
            mock_calibrate.return_value = (
                0.25,  # RMS error
                np.array([[850, 0, 320], [0, 850, 240], [0, 0, 1]], dtype=float),
                np.array([-0.08, 0.03, 0, 0, 0], dtype=float),
                None,  # rvecs
                None   # tvecs
            )
            
            calibrator.images = mock_images
            calibrator.image_size = (640, 480)
            
            # Run calibration
            rms_error = calibrator.calibrate_camera_charuco()
            
            # Verify results
            assert rms_error == 0.25
            assert calibrator.camera_matrix is not None
            assert calibrator.distortion_coefficients is not None

class TestEyeInHandCalibrationWorkflow:
    """Test complete eye-in-hand calibration workflows."""
    
    @pytest.mark.integration
    def test_complete_eye_in_hand_workflow(self, temp_output_dir):
        """Test complete eye-in-hand calibration workflow."""
        # Set up calibration data directory
        calib_dir = temp_output_dir / "eye_in_hand_data"
        calib_dir.mkdir()
        
        # Create pattern configuration
        pattern = create_standard_chessboard(rows=7, columns=10, square_size=20.0)
        pattern_file = calib_dir / "chessboard_config.json"
        save_pattern_to_json(pattern, str(pattern_file))
        
        # Create mock image files
        image_files = ['img001.jpg', 'img002.jpg', 'img003.jpg', 'img004.jpg', 'img005.jpg']
        for img_file in image_files:
            (calib_dir / img_file).touch()
        
        # Create corresponding pose files
        pose_data_list = []
        for i, img_file in enumerate(image_files):
            pose_data = {
                'end_xyzrpy': [0.1 + 0.1*i, 0.2 + 0.1*i, 0.3 + 0.1*i, 0, 0, 0.1*i],
                'end2base': [
                    [1, 0, 0, 0.1 + 0.1*i],
                    [0, 1, 0, 0.2 + 0.1*i], 
                    [0, 0, 1, 0.3 + 0.1*i],
                    [0, 0, 0, 1]
                ]
            }
            pose_data_list.append(pose_data)
            
            pose_file = calib_dir / f"pose{i+1:03d}.json"
            with open(pose_file, 'w') as f:
                json.dump(pose_data, f)
        
        # Create eye-in-hand calibrator
        mock_camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
        mock_dist_coeffs = np.array([-0.1, 0.05, 0, 0, 0], dtype=float)
        
        calibrator = EyeInHandCalibrator(
            camera_matrix=mock_camera_matrix,
            distortion_coefficients=mock_dist_coeffs
        )
        
        # Mock the image loading and pattern detection
        with patch('cv2.imread') as mock_imread, \
             patch('cv2.findChessboardCorners') as mock_find_corners, \
             patch('cv2.solvePnP') as mock_solve_pnp, \
             patch('cv2.calibrateHandEye') as mock_calibrate_hand_eye:
            
            # Mock image loading
            mock_imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Mock successful corner detection
            mock_corners = np.random.rand(70, 1, 2).astype(np.float32)
            mock_find_corners.return_value = (True, mock_corners)
            
            # Mock successful PnP solution
            mock_solve_pnp.return_value = (
                True,
                np.random.rand(3, 1),  # rvec
                np.random.rand(3, 1)   # tvec
            )
            
            # Mock successful hand-eye calibration
            mock_R = np.eye(3)
            mock_t = np.array([[0.05], [0.02], [0.1]])
            mock_calibrate_hand_eye.return_value = (mock_R, mock_t)
            
            # Load calibration data and perform calibration
            calibrator.load_calibration_data(str(calib_dir))
            
            # Verify data was loaded correctly
            assert len(calibrator.image_paths) == 5
            assert len(calibrator.robot_poses) == 5
            
            # Perform the calibration
            R, t = calibrator.calibrate()
            
            # Verify results
            assert R is not None
            assert t is not None
            assert R.shape == (3, 3)
            assert t.shape == (3, 1)
            
            # Test saving results
            results_file = temp_output_dir / "hand_eye_results.json"
            calibrator.save_calibration_results(str(results_file))
            assert results_file.exists()

class TestPatternRoundTripWorkflow:
    """Test pattern creation, saving, and loading workflows."""
    
    @pytest.mark.integration
    def test_pattern_roundtrip_workflow(self, temp_output_dir):
        """Test complete pattern creation, save, and load workflow."""
        patterns_to_test = [
            ('chessboard', lambda: create_standard_chessboard(7, 10, 20.0)),
            ('charuco', lambda: create_charuco_board(8, 6, 40.0, 30.0, cv2.aruco.DICT_6X6_250)),
            ('aruco_grid', lambda: create_aruco_grid_board(4, 3, 50.0, 10.0, cv2.aruco.DICT_4X4_50))
        ]
        
        for pattern_name, pattern_creator in patterns_to_test:
            with pytest.raises(Exception) if pattern_name == 'invalid' else None:
                # Create pattern
                original_pattern = pattern_creator()
                
                # Save pattern
                pattern_file = temp_output_dir / f"{pattern_name}_config.json"
                save_pattern_to_json(original_pattern, str(pattern_file))
                
                # Verify file was created
                assert pattern_file.exists()
                
                # Load pattern back
                loaded_pattern = load_pattern_from_json(str(pattern_file))
                
                # Verify pattern properties are preserved
                assert loaded_pattern['pattern_type'] == original_pattern['pattern_type']
                
                # Pattern-specific verification
                if pattern_name == 'chessboard':
                    assert loaded_pattern['rows'] == original_pattern['rows']
                    assert loaded_pattern['columns'] == original_pattern['columns']
                    assert loaded_pattern['square_size'] == original_pattern['square_size']
                
                elif pattern_name == 'charuco':
                    assert loaded_pattern['squares_x'] == original_pattern['squares_x']
                    assert loaded_pattern['squares_y'] == original_pattern['squares_y']
                    assert loaded_pattern['square_length'] == original_pattern['square_length']
                    assert loaded_pattern['marker_length'] == original_pattern['marker_length']
                
                elif pattern_name == 'aruco_grid':
                    assert loaded_pattern['markers_x'] == original_pattern['markers_x']
                    assert loaded_pattern['markers_y'] == original_pattern['markers_y']
                    assert loaded_pattern['marker_length'] == original_pattern['marker_length']
                    assert loaded_pattern['marker_separation'] == original_pattern['marker_separation']

class TestCalibrationDataPersistence:
    """Test calibration data saving and loading workflows."""
    
    @pytest.mark.integration
    def test_calibration_results_persistence(self, temp_output_dir):
        """Test saving and loading calibration results."""
        # Create calibrator with mock results
        calibrator = IntrinsicCalibrator()
        calibrator.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
        calibrator.distortion_coefficients = np.array([-0.1, 0.05, 0, 0, 0], dtype=float)
        calibrator.rms_error = 0.35
        
        # Save calibration results
        results_file = temp_output_dir / "calibration_results.json"
        calibrator.save_calibration(str(results_file))
        
        # Verify file exists and has correct structure
        assert results_file.exists()
        
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        assert 'camera_matrix' in results_data
        assert 'distortion_coefficients' in results_data
        assert 'rms_error' in results_data
        assert results_data['rms_error'] == 0.35
        
        # Verify matrix data is properly serialized
        loaded_camera_matrix = np.array(results_data['camera_matrix'])
        np.testing.assert_array_almost_equal(loaded_camera_matrix, calibrator.camera_matrix)
    
    @pytest.mark.integration
    def test_hand_eye_results_persistence(self, temp_output_dir):
        """Test saving and loading hand-eye calibration results."""
        calibrator = EyeInHandCalibrator()
        
        # Mock calibration results
        R = np.eye(3)
        t = np.array([[0.05], [0.02], [0.1]])
        
        # Save results
        results_file = temp_output_dir / "hand_eye_results.json"
        calibrator.save_calibration_results(str(results_file), R, t)
        
        # Verify file exists
        assert results_file.exists()
        
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        assert 'rotation_matrix' in results_data
        assert 'translation_vector' in results_data
        assert 'transformation_matrix' in results_data
        
        # Verify data integrity
        loaded_R = np.array(results_data['rotation_matrix'])
        loaded_t = np.array(results_data['translation_vector'])
        
        np.testing.assert_array_almost_equal(loaded_R, R)
        np.testing.assert_array_almost_equal(loaded_t, t.flatten())

class TestErrorHandlingWorkflows:
    """Test error handling in complete workflows."""
    
    @pytest.mark.integration
    def test_insufficient_data_workflow(self, temp_output_dir):
        """Test workflow behavior with insufficient calibration data."""
        # Create directory with too few images
        calib_dir = temp_output_dir / "insufficient_data"
        calib_dir.mkdir()
        
        # Only create 2 images (insufficient for calibration)
        for i in range(2):
            (calib_dir / f"img{i:03d}.jpg").touch()
            pose_data = {'end_xyzrpy': [0.1*i, 0.2*i, 0.3*i, 0, 0, 0]}
            with open(calib_dir / f"pose{i:03d}.json", 'w') as f:
                json.dump(pose_data, f)
        
        calibrator = EyeInHandCalibrator()
        
        # Should handle insufficient data gracefully
        with pytest.raises(ValueError, match="Insufficient"):
            calibrator.load_calibration_data(str(calib_dir))
            calibrator.calibrate()
    
    @pytest.mark.integration
    def test_corrupted_data_workflow(self, temp_output_dir):
        """Test workflow behavior with corrupted data files."""
        # Create directory with corrupted data
        calib_dir = temp_output_dir / "corrupted_data"
        calib_dir.mkdir()
        
        # Create valid image files
        for i in range(5):
            (calib_dir / f"img{i:03d}.jpg").touch()
        
        # Create mix of valid and corrupted pose files
        for i in range(3):
            pose_data = {'end_xyzrpy': [0.1*i, 0.2*i, 0.3*i, 0, 0, 0]}
            with open(calib_dir / f"pose{i:03d}.json", 'w') as f:
                json.dump(pose_data, f)
        
        # Corrupted pose files
        with open(calib_dir / "pose003.json", 'w') as f:
            f.write("invalid json content")
        
        with open(calib_dir / "pose004.json", 'w') as f:
            json.dump({'invalid_field': 'invalid_data'}, f)
        
        calibrator = EyeInHandCalibrator()
        
        # Should handle corrupted data gracefully
        with pytest.raises((json.JSONDecodeError, KeyError, ValueError)):
            calibrator.load_calibration_data(str(calib_dir))

class TestConcurrentWorkflows:
    """Test multiple calibration workflows running concurrently."""
    
    @pytest.mark.integration
    def test_multiple_calibrator_instances(self, temp_output_dir):
        """Test multiple calibrator instances working independently."""
        # Create multiple calibrators
        calibrator1 = IntrinsicCalibrator()
        calibrator2 = IntrinsicCalibrator()
        
        # Set different parameters
        pattern1 = create_standard_chessboard(7, 10, 20.0)
        pattern2 = create_standard_chessboard(8, 11, 25.0)
        
        calibrator1.calibration_pattern = pattern1
        calibrator1.pattern_type = 'standard_chessboard'
        
        calibrator2.calibration_pattern = pattern2
        calibrator2.pattern_type = 'standard_chessboard'
        
        # Verify they remain independent
        assert calibrator1.calibration_pattern['rows'] == 7
        assert calibrator2.calibration_pattern['rows'] == 8
        
        assert calibrator1.calibration_pattern['square_size'] == 20.0
        assert calibrator2.calibration_pattern['square_size'] == 25.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
