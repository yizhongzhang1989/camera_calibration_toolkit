"""
Unit tests for eye-in-hand calibration functionality.

Tests the eye-in-hand calibration logic, data loading,
and transformation calculations without requiring full
calibration workflows or real data files.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from core.eye_in_hand_calibration import EyeInHandCalibrator

class TestEyeInHandDataLoading:
    """Test data loading functionality for eye-in-hand calibration."""
    
    @pytest.mark.unit
    def test_load_calibration_data_basic(self, temp_output_dir):
        """Test basic calibration data loading."""
        # Create mock files
        image_files = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        pose_files = ['pose1.json', 'pose2.json', 'pose3.json']
        config_file = 'chessboard_config.json'
        
        # Create actual files
        for img_file in image_files:
            (temp_output_dir / img_file).touch()
        
        for pose_file in pose_files:
            pose_data = {
                'end_xyzrpy': [0.1, 0.2, 0.3, 0, 0, 0],
                'end2base': [[1,0,0,0.1],[0,1,0,0.2],[0,0,1,0.3],[0,0,0,1]]
            }
            with open(temp_output_dir / pose_file, 'w') as f:
                json.dump(pose_data, f)
        
        # Create config file (should be excluded)
        config_data = {'pattern_type': 'chessboard', 'rows': 7, 'columns': 10}
        with open(temp_output_dir / config_file, 'w') as f:
            json.dump(config_data, f)
        
        calibrator = EyeInHandCalibrator()
        
        with patch('core.eye_in_hand_calibration.os.listdir') as mock_listdir, \
             patch('core.eye_in_hand_calibration.os.path.isfile') as mock_isfile:
            
            all_files = image_files + pose_files + [config_file]
            mock_listdir.return_value = all_files
            mock_isfile.return_value = True
            
            # Mock the actual loading functions
            with patch.object(calibrator, '_load_pose_data') as mock_load_pose, \
                 patch.object(calibrator, '_load_image_paths') as mock_load_images:
                
                mock_load_pose.return_value = [{'end_xyzrpy': [0.1, 0.2, 0.3, 0, 0, 0]} for _ in range(3)]
                mock_load_images.return_value = image_files
                
                result = calibrator.load_calibration_data(str(temp_output_dir))
                
                # Should exclude config file from pose files
                # Only 3 pose files should be loaded (config file excluded)
                mock_load_pose.assert_called_once()
                mock_load_images.assert_called_once()
    
    @pytest.mark.unit
    def test_load_calibration_data_filters_config_files(self, temp_output_dir):
        """Test that config files are properly filtered out from pose files."""
        # Create files including config files that should be excluded
        image_files = ['img1.jpg', 'img2.jpg']  
        pose_files = ['pose1.json', 'pose2.json']
        config_files = ['chessboard_config.json', 'calibration_config.json', 'pattern_config.json']
        
        for file_list in [image_files, pose_files, config_files]:
            for file_name in file_list:
                (temp_output_dir / file_name).touch()
        
        calibrator = EyeInHandCalibrator()
        
        # Test the internal filter method
        all_files = image_files + pose_files + config_files
        filtered_pose_files = calibrator._filter_pose_files(all_files)
        
        # Should only include actual pose files, not config files
        expected_pose_files = pose_files  # Only pose1.json, pose2.json
        assert len(filtered_pose_files) == len(expected_pose_files)
        
        for pose_file in expected_pose_files:
            assert pose_file in filtered_pose_files
        
        for config_file in config_files:
            assert config_file not in filtered_pose_files
    
    @pytest.mark.unit
    def test_validate_data_consistency(self):
        """Test validation of image-pose data consistency."""
        calibrator = EyeInHandCalibrator()
        
        # Consistent data
        images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        poses = [
            {'end_xyzrpy': [0.1, 0.2, 0.3, 0, 0, 0]},
            {'end_xyzrpy': [0.2, 0.3, 0.4, 0, 0, 0]},
            {'end_xyzrpy': [0.3, 0.4, 0.5, 0, 0, 0]}
        ]
        
        assert calibrator._validate_data_consistency(images, poses)
        
        # Inconsistent data (more poses than images)
        poses.append({'end_xyzrpy': [0.4, 0.5, 0.6, 0, 0, 0]})
        assert not calibrator._validate_data_consistency(images, poses)
        
        # Inconsistent data (more images than poses)
        images.append('img4.jpg')
        images.append('img5.jpg')
        assert not calibrator._validate_data_consistency(images, poses)

class TestRobotPoseProcessing:
    """Test robot pose data processing and transformation."""
    
    @pytest.mark.unit
    def test_parse_robot_pose_xyzrpy_format(self):
        """Test parsing robot poses in XYZ+RPY format."""
        calibrator = EyeInHandCalibrator()
        
        pose_data = {
            'end_xyzrpy': [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]  # x,y,z,roll,pitch,yaw
        }
        
        transformation_matrix = calibrator._parse_robot_pose(pose_data)
        
        assert transformation_matrix is not None
        assert transformation_matrix.shape == (4, 4)
        
        # Check translation components
        assert abs(transformation_matrix[0, 3] - 0.1) < 1e-6  # x
        assert abs(transformation_matrix[1, 3] - 0.2) < 1e-6  # y  
        assert abs(transformation_matrix[2, 3] - 0.3) < 1e-6  # z
        
        # Check that it's a proper transformation matrix
        assert abs(transformation_matrix[3, 0]) < 1e-6
        assert abs(transformation_matrix[3, 1]) < 1e-6
        assert abs(transformation_matrix[3, 2]) < 1e-6
        assert abs(transformation_matrix[3, 3] - 1.0) < 1e-6
    
    @pytest.mark.unit
    def test_parse_robot_pose_matrix_format(self):
        """Test parsing robot poses in transformation matrix format."""
        calibrator = EyeInHandCalibrator()
        
        # 4x4 transformation matrix
        matrix = [[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]]
        pose_data = {
            'end2base': matrix
        }
        
        transformation_matrix = calibrator._parse_robot_pose(pose_data)
        
        assert transformation_matrix is not None
        assert transformation_matrix.shape == (4, 4)
        np.testing.assert_array_almost_equal(transformation_matrix, np.array(matrix))
    
    @pytest.mark.unit
    def test_convert_multiple_poses(self):
        """Test converting multiple robot poses to transformation matrices."""
        calibrator = EyeInHandCalibrator()
        
        poses_data = [
            {'end_xyzrpy': [0.1, 0.2, 0.3, 0, 0, 0]},
            {'end_xyzrpy': [0.2, 0.3, 0.4, 0, 0, 0.1]},
            {'end_xyzrpy': [0.3, 0.4, 0.5, 0, 0.1, 0]}
        ]
        
        transformation_matrices = calibrator._convert_poses_to_matrices(poses_data)
        
        assert len(transformation_matrices) == 3
        for matrix in transformation_matrices:
            assert matrix.shape == (4, 4)
            # Verify it's a proper transformation matrix
            assert abs(matrix[3, 3] - 1.0) < 1e-6

class TestCameraPoseEstimation:
    """Test camera pose estimation from calibration patterns."""
    
    @pytest.mark.unit
    @patch('cv2.solvePnP')
    def test_estimate_camera_poses_success(self, mock_solve_pnp):
        """Test successful camera pose estimation."""
        # Mock successful PnP solution
        mock_solve_pnp.return_value = (
            True,  # Success
            np.array([[0.1], [0.2], [0.3]]),  # rvec
            np.array([[0.1], [0.2], [0.3]])   # tvec
        )
        
        calibrator = EyeInHandCalibrator()
        calibrator.camera_matrix = np.eye(3)
        calibrator.distortion_coefficients = np.zeros(5)
        
        # Mock detection results
        object_points = [np.random.rand(20, 3) for _ in range(3)]
        image_points = [np.random.rand(20, 2) for _ in range(3)]
        
        camera_poses = calibrator._estimate_camera_poses(object_points, image_points)
        
        assert len(camera_poses) == 3
        for pose in camera_poses:
            assert pose.shape == (4, 4)
            
        # Verify PnP was called for each image
        assert mock_solve_pnp.call_count == 3
    
    @pytest.mark.unit
    @patch('cv2.solvePnP')
    def test_estimate_camera_poses_failure(self, mock_solve_pnp):
        """Test handling of camera pose estimation failures."""
        # Mock failed PnP solution
        mock_solve_pnp.return_value = (
            False,  # Failure
            None,
            None
        )
        
        calibrator = EyeInHandCalibrator()
        calibrator.camera_matrix = np.eye(3)
        calibrator.distortion_coefficients = np.zeros(5)
        
        object_points = [np.random.rand(20, 3)]
        image_points = [np.random.rand(20, 2)]
        
        with pytest.raises(ValueError, match="Failed to estimate camera pose"):
            calibrator._estimate_camera_poses(object_points, image_points)

class TestHandEyeCalibration:
    """Test hand-eye calibration algorithm integration."""
    
    @pytest.mark.unit
    @patch('cv2.calibrateHandEye')
    def test_hand_eye_calibration_success(self, mock_calibrate_hand_eye):
        """Test successful hand-eye calibration."""
        # Mock successful hand-eye calibration
        mock_R = np.eye(3)
        mock_t = np.array([[0.1], [0.2], [0.3]])
        mock_calibrate_hand_eye.return_value = (mock_R, mock_t)
        
        calibrator = EyeInHandCalibrator()
        
        # Mock input data
        robot_poses = [np.eye(4) for _ in range(3)]
        camera_poses = [np.eye(4) for _ in range(3)]
        
        R, t = calibrator._perform_hand_eye_calibration(robot_poses, camera_poses)
        
        assert R is not None
        assert t is not None
        assert R.shape == (3, 3)
        assert t.shape == (3, 1)
        
        mock_calibrate_hand_eye.assert_called_once()
    
    @pytest.mark.unit
    def test_transform_extraction_from_matrices(self):
        """Test extracting rotation and translation from transformation matrices."""
        calibrator = EyeInHandCalibrator()
        
        # Create test transformation matrices
        matrices = []
        for i in range(3):
            matrix = np.eye(4)
            matrix[0, 3] = 0.1 * i  # x translation
            matrix[1, 3] = 0.2 * i  # y translation
            matrix[2, 3] = 0.3 * i  # z translation
            matrices.append(matrix)
        
        R_list, t_list = calibrator._extract_rotations_translations(matrices)
        
        assert len(R_list) == 3
        assert len(t_list) == 3
        
        for R, t in zip(R_list, t_list):
            assert R.shape == (3, 3)
            assert t.shape == (3, 1)

class TestCalibrationResultValidation:
    """Test validation of hand-eye calibration results."""
    
    @pytest.mark.unit
    def test_validate_hand_eye_results_valid(self):
        """Test validation of valid hand-eye calibration results."""
        calibrator = EyeInHandCalibrator()
        
        # Valid rotation matrix (orthogonal, determinant = 1)
        R = np.eye(3)
        t = np.array([[0.1], [0.2], [0.3]])
        
        assert calibrator._validate_hand_eye_results(R, t)
    
    @pytest.mark.unit
    def test_validate_hand_eye_results_invalid_rotation(self):
        """Test validation with invalid rotation matrix."""
        calibrator = EyeInHandCalibrator()
        
        # Invalid rotation matrix (not orthogonal)
        R = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
        t = np.array([[0.1], [0.2], [0.3]])
        
        assert not calibrator._validate_hand_eye_results(R, t)
    
    @pytest.mark.unit
    def test_validate_hand_eye_results_extreme_translation(self):
        """Test validation with extreme translation values."""
        calibrator = EyeInHandCalibrator()
        
        R = np.eye(3)
        
        # Reasonable translation
        t_reasonable = np.array([[0.1], [0.2], [0.3]])
        assert calibrator._validate_hand_eye_results(R, t_reasonable)
        
        # Extreme translation (should be rejected)
        t_extreme = np.array([[100.0], [200.0], [300.0]])
        assert not calibrator._validate_hand_eye_results(R, t_extreme)

class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in eye-in-hand calibration."""
    
    @pytest.mark.unit
    def test_insufficient_data_poses(self):
        """Test handling of insufficient pose data."""
        calibrator = EyeInHandCalibrator()
        
        # Too few poses
        robot_poses = [np.eye(4)]  # Only 1 pose
        camera_poses = [np.eye(4)]
        
        with pytest.raises(ValueError, match="Insufficient data"):
            calibrator._perform_hand_eye_calibration(robot_poses, camera_poses)
    
    @pytest.mark.unit
    def test_mismatched_pose_counts(self):
        """Test handling of mismatched robot and camera pose counts."""
        calibrator = EyeInHandCalibrator()
        
        robot_poses = [np.eye(4) for _ in range(5)]   # 5 poses
        camera_poses = [np.eye(4) for _ in range(3)]  # 3 poses
        
        with pytest.raises(ValueError, match="Mismatched number"):
            calibrator._perform_hand_eye_calibration(robot_poses, camera_poses)
    
    @pytest.mark.unit
    def test_corrupted_pose_data(self):
        """Test handling of corrupted pose data."""
        calibrator = EyeInHandCalibrator()
        
        # Corrupted robot pose data
        corrupted_pose = {
            'invalid_field': [1, 2, 3]  # Missing required fields
        }
        
        with pytest.raises(KeyError):
            calibrator._parse_robot_pose(corrupted_pose)
    
    @pytest.mark.unit
    def test_numerical_stability_edge_cases(self):
        """Test numerical stability with edge case inputs."""
        calibrator = EyeInHandCalibrator()
        
        # Very small translation values
        t_small = np.array([[1e-10], [1e-10], [1e-10]])
        R = np.eye(3)
        
        # Should still validate
        assert calibrator._validate_hand_eye_results(R, t_small)
        
        # NaN values should be rejected
        t_nan = np.array([[np.nan], [0.1], [0.2]])
        assert not calibrator._validate_hand_eye_results(R, t_nan)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
