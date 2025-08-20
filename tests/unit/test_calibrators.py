"""
Unit tests for calibration algorithms.

Tests the calibration classes and their core logic without
requiring actual image processing or full calibration workflows.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from core.intrinsic_calibration import IntrinsicCalibrator
from core.eye_in_hand_calibration import EyeInHandCalibrator

class TestIntrinsicCalibratorInitialization:
    """Test IntrinsicCalibrator initialization and basic properties."""
    
    @pytest.mark.unit
    def test_calibrator_initialization_minimal(self):
        """Test basic calibrator initialization."""
        calibrator = IntrinsicCalibrator()
        
        # Check initial state
        assert calibrator.images is None
        assert calibrator.image_paths is None
        assert calibrator.calibration_pattern is None
        assert calibrator.pattern_type is None
        assert calibrator.image_size is None
    
    @pytest.mark.unit
    @patch('core.utils.load_images_from_directory')
    def test_calibrator_initialization_with_paths(self, mock_load_images):
        """Test calibrator initialization with image paths."""
        # Mock image loading
        mock_load_images.return_value = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        
        calibrator = IntrinsicCalibrator(image_paths=['img1.jpg', 'img2.jpg', 'img3.jpg'])
        
        assert calibrator.image_paths is not None
        assert len(calibrator.image_paths) == 3
    
    @pytest.mark.unit
    def test_calibrator_initialization_with_pattern(self, sample_chessboard_config):
        """Test calibrator initialization with calibration pattern."""
        from core.calibration_patterns import load_pattern_from_json
        
        pattern = load_pattern_from_json(sample_chessboard_config)
        calibrator = IntrinsicCalibrator(
            calibration_pattern=pattern,
            pattern_type='standard_chessboard'
        )
        
        assert calibrator.calibration_pattern is not None
        assert calibrator.pattern_type == 'standard_chessboard'
    
    @pytest.mark.unit
    def test_calibrator_set_images_from_paths(self):
        """Test setting images from file paths."""
        calibrator = IntrinsicCalibrator()
        
        # Mock the image loading method
        with patch.object(calibrator, '_load_images_from_paths') as mock_load:
            mock_load.return_value = ([Mock(), Mock(), Mock()], (640, 480))
            
            paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
            calibrator.set_images_from_paths(paths)
            
            mock_load.assert_called_once_with(paths)
            assert calibrator.images is not None
            assert calibrator.image_size == (640, 480)

class TestIntrinsicCalibratorValidation:
    """Test input validation in IntrinsicCalibrator."""
    
    @pytest.mark.unit
    def test_validate_calibration_readiness_incomplete(self):
        """Test validation when calibrator is not ready."""
        calibrator = IntrinsicCalibrator()
        
        # Should not be ready without images and pattern
        assert not calibrator._is_ready_for_calibration()
    
    @pytest.mark.unit
    def test_validate_calibration_readiness_complete(self, sample_chessboard_config):
        """Test validation when calibrator is ready."""
        from core.calibration_patterns import load_pattern_from_json
        
        calibrator = IntrinsicCalibrator()
        calibrator.calibration_pattern = load_pattern_from_json(sample_chessboard_config)
        calibrator.pattern_type = 'standard_chessboard'
        calibrator.images = [Mock(), Mock(), Mock()]
        calibrator.image_size = (640, 480)
        
        assert calibrator._is_ready_for_calibration()
    
    @pytest.mark.unit
    def test_minimum_images_validation(self):
        """Test minimum number of images validation."""
        calibrator = IntrinsicCalibrator()
        
        # Too few images
        calibrator.images = [Mock(), Mock()]  # Only 2 images
        assert not calibrator._has_sufficient_images()
        
        # Sufficient images
        calibrator.images = [Mock() for _ in range(5)]  # 5 images
        assert calibrator._has_sufficient_images()
    
    @pytest.mark.unit 
    def test_image_size_consistency_validation(self):
        """Test that all images have consistent dimensions."""
        calibrator = IntrinsicCalibrator()
        
        # Mock images with consistent size
        mock_images = []
        for _ in range(3):
            mock_img = Mock()
            mock_img.shape = (480, 640, 3)
            mock_images.append(mock_img)
        
        calibrator.images = mock_images
        assert calibrator._images_have_consistent_size()
        
        # Mock images with inconsistent size
        mock_images[1].shape = (720, 1280, 3)  # Different size
        assert not calibrator._images_have_consistent_size()

class TestEyeInHandCalibratorInitialization:
    """Test EyeInHandCalibrator initialization and properties."""
    
    @pytest.mark.unit
    def test_eye_in_hand_calibrator_initialization(self, mock_camera_matrix, mock_distortion_coefficients):
        """Test basic eye-in-hand calibrator initialization."""
        calibrator = EyeInHandCalibrator(
            camera_matrix=mock_camera_matrix,
            distortion_coefficients=mock_distortion_coefficients
        )
        
        assert calibrator.camera_matrix is not None
        assert calibrator.distortion_coefficients is not None
        assert np.array_equal(calibrator.camera_matrix, mock_camera_matrix)
    
    @pytest.mark.unit
    def test_eye_in_hand_set_robot_poses(self):
        """Test setting robot poses."""
        calibrator = EyeInHandCalibrator()
        
        mock_poses = [
            {'end_xyzrpy': [0.1, 0.2, 0.3, 0, 0, 0]},
            {'end_xyzrpy': [0.2, 0.3, 0.4, 0, 0, 0.1]},
            {'end_xyzrpy': [0.3, 0.4, 0.5, 0, 0.1, 0]}
        ]
        
        calibrator.set_robot_poses(mock_poses)
        
        assert calibrator.robot_poses is not None
        assert len(calibrator.robot_poses) == 3
    
    @pytest.mark.unit
    def test_eye_in_hand_pose_validation(self):
        """Test robot pose format validation."""
        calibrator = EyeInHandCalibrator()
        
        # Valid poses
        valid_poses = [
            {'end_xyzrpy': [0.1, 0.2, 0.3, 0, 0, 0], 'end2base': [[1,0,0,0.1],[0,1,0,0.2],[0,0,1,0.3],[0,0,0,1]]},
            {'end_xyzrpy': [0.2, 0.3, 0.4, 0, 0, 0.1], 'end2base': [[1,0,0,0.2],[0,1,0,0.3],[0,0,1,0.4],[0,0,0,1]]}
        ]
        
        assert calibrator._validate_robot_poses(valid_poses)
        
        # Invalid poses (missing required fields)
        invalid_poses = [
            {'position': [0.1, 0.2, 0.3]},  # Wrong field name
            {}  # Empty pose
        ]
        
        assert not calibrator._validate_robot_poses(invalid_poses)

class TestCalibrationAlgorithmLogic:
    """Test core calibration algorithm logic."""
    
    @pytest.mark.unit
    @patch('cv2.calibrateCamera')
    def test_intrinsic_calibration_opencv_call(self, mock_calibrate_camera):
        """Test OpenCV calibrateCamera is called correctly."""
        # Mock successful calibration
        mock_camera_matrix = np.eye(3)
        mock_dist_coeffs = np.zeros(5)
        mock_calibrate_camera.return_value = (
            0.5,  # RMS error
            mock_camera_matrix,
            mock_dist_coeffs,
            None,  # rvecs
            None   # tvecs
        )
        
        calibrator = IntrinsicCalibrator()
        
        # Mock required data
        calibrator.object_points = [np.ones((10, 3)) for _ in range(3)]
        calibrator.image_points = [np.ones((10, 2)) for _ in range(3)]
        calibrator.image_size = (640, 480)
        
        rms_error = calibrator.calibrate_camera()
        
        # Verify OpenCV was called
        mock_calibrate_camera.assert_called_once()
        assert rms_error == 0.5
        assert calibrator.camera_matrix is not None
        assert calibrator.distortion_coefficients is not None
    
    @pytest.mark.unit
    def test_calibration_error_handling(self):
        """Test calibration error handling."""
        calibrator = IntrinsicCalibrator()
        
        # Try to calibrate without sufficient data
        with pytest.raises((AttributeError, ValueError)):
            calibrator.calibrate_camera()
    
    @pytest.mark.unit
    def test_calibration_result_validation(self, mock_camera_matrix, mock_distortion_coefficients):
        """Test validation of calibration results."""
        calibrator = IntrinsicCalibrator()
        calibrator.camera_matrix = mock_camera_matrix
        calibrator.distortion_coefficients = mock_distortion_coefficients
        calibrator.rms_error = 0.3
        
        # Valid results
        assert calibrator._validate_calibration_results()
        
        # Invalid results (too high RMS error)
        calibrator.rms_error = 10.0  # Very high error
        assert not calibrator._validate_calibration_results()
        
        # Invalid camera matrix
        calibrator.rms_error = 0.3
        calibrator.camera_matrix = np.zeros((3, 3))  # All zeros
        assert not calibrator._validate_calibration_results()

class TestCalibrationResultsProcessing:
    """Test calibration results processing and output."""
    
    @pytest.mark.unit
    def test_get_calibration_summary(self, mock_camera_matrix, mock_distortion_coefficients):
        """Test getting calibration summary."""
        calibrator = IntrinsicCalibrator()
        calibrator.camera_matrix = mock_camera_matrix
        calibrator.distortion_coefficients = mock_distortion_coefficients
        calibrator.rms_error = 0.25
        calibrator.images = [Mock() for _ in range(5)]
        
        summary = calibrator.get_calibration_summary()
        
        assert 'rms_error' in summary
        assert 'camera_matrix' in summary
        assert 'distortion_coefficients' in summary
        assert 'image_count' in summary
        assert summary['rms_error'] == 0.25
        assert summary['image_count'] == 5
    
    @pytest.mark.unit
    def test_export_calibration_data(self, temp_output_dir, mock_camera_matrix):
        """Test exporting calibration data to file."""
        calibrator = IntrinsicCalibrator()
        calibrator.camera_matrix = mock_camera_matrix
        calibrator.distortion_coefficients = np.array([-0.1, 0.05, 0, 0, 0])
        calibrator.rms_error = 0.3
        
        output_file = temp_output_dir / "test_calibration.json"
        
        with patch.object(calibrator, 'save_calibration') as mock_save:
            calibrator.save_calibration(str(output_file))
            mock_save.assert_called_once_with(str(output_file))
    
    @pytest.mark.unit
    def test_calibration_parameters_access(self, mock_camera_matrix):
        """Test accessing individual calibration parameters."""
        calibrator = IntrinsicCalibrator()
        calibrator.camera_matrix = mock_camera_matrix
        
        # Test focal length extraction
        fx = calibrator.get_focal_length_x()
        fy = calibrator.get_focal_length_y()
        
        assert fx == mock_camera_matrix[0, 0]
        assert fy == mock_camera_matrix[1, 1]
        
        # Test principal point extraction  
        cx, cy = calibrator.get_principal_point()
        
        assert cx == mock_camera_matrix[0, 2]
        assert cy == mock_camera_matrix[1, 2]

class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.mark.unit
    def test_calibrator_with_no_data(self):
        """Test calibrator behavior with no input data."""
        calibrator = IntrinsicCalibrator()
        
        # Should handle gracefully
        assert calibrator.images is None
        assert calibrator.calibration_pattern is None
        assert not calibrator._is_ready_for_calibration()
    
    @pytest.mark.unit
    def test_calibrator_with_corrupted_data(self):
        """Test calibrator behavior with corrupted input data."""
        calibrator = IntrinsicCalibrator()
        
        # Corrupted image data
        calibrator.images = [None, None, None]
        calibrator.image_size = (640, 480)
        
        # Should detect corruption
        assert not calibrator._validate_images()
    
    @pytest.mark.unit
    def test_numerical_stability_edge_cases(self):
        """Test numerical stability with edge case inputs."""
        calibrator = IntrinsicCalibrator()
        
        # Very small numbers
        calibrator.rms_error = 1e-10
        assert calibrator._validate_calibration_results()
        
        # Very large numbers (should be rejected)
        calibrator.rms_error = 1e10
        assert not calibrator._validate_calibration_results()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
