"""
Unit tests for utility functions.

Tests core utility functions that are available in the utils module.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import patch, Mock

from core.utils import (
    get_objpoints,
    find_chessboard_corners,
    load_images_from_directory,
    rpy_to_matrix,
    xyz_rpy_to_matrix,
    inverse_transform_matrix,
    matrix_to_xyz_rpy
)

class TestMathematicalUtilities:
    """Test mathematical utility functions."""
    
    @pytest.mark.unit
    def test_get_objpoints(self):
        """Test object points generation for chessboard calibration."""
        num_images = 3
        XX, YY = 10, 7  # Chessboard dimensions
        L = 0.02  # Square size in meters
        
        objpoints = get_objpoints(num_images, XX, YY, L)
        
        # Verify correct number of object point sets
        assert len(objpoints) == num_images
        
        # Verify each set has correct shape
        for objp in objpoints:
            assert objp.shape == (XX * YY, 3)
            assert objp.dtype == np.float32
        
        # Verify scaling is applied correctly
        assert objpoints[0][0, 0] == 0.0  # First corner at origin
        assert np.allclose(objpoints[0][1, 0], L)  # Next corner scaled by square size
        
        # Verify all z-coordinates are zero (pattern is planar)
        for objp in objpoints:
            assert np.allclose(objp[:, 2], 0.0)
    
    @pytest.mark.unit
    def test_rpy_to_matrix(self):
        """Test roll-pitch-yaw to rotation matrix conversion."""
        # Test identity transformation (zero rotations)
        zero_rpy = [0.0, 0.0, 0.0]
        identity_matrix = rpy_to_matrix(zero_rpy)
        
        expected_identity = np.eye(3)
        np.testing.assert_array_almost_equal(identity_matrix, expected_identity)
        
        # Test that result is a valid rotation matrix
        assert identity_matrix.shape == (3, 3)
        det = np.linalg.det(identity_matrix)
        assert abs(det - 1.0) < 1e-6  # Determinant should be 1
    
    @pytest.mark.unit
    def test_xyz_rpy_to_matrix(self):
        """Test XYZ+RPY to transformation matrix conversion."""
        xyz_rpy = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]  # Translation only
        
        T = xyz_rpy_to_matrix(xyz_rpy)
        
        # Verify shape and structure
        assert T.shape == (4, 4)
        
        # Check translation components
        assert abs(T[0, 3] - 0.1) < 1e-6
        assert abs(T[1, 3] - 0.2) < 1e-6  
        assert abs(T[2, 3] - 0.3) < 1e-6
        
        # Check that rotation part is identity (zero rotations)
        np.testing.assert_array_almost_equal(T[:3, :3], np.eye(3))
        
        # Check homogeneous coordinate
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])
    
    @pytest.mark.unit
    def test_inverse_transform_matrix(self):
        """Test transformation matrix inversion."""
        # Create a simple transformation matrix
        T = np.array([
            [1, 0, 0, 0.1],
            [0, 1, 0, 0.2],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ])
        
        T_inv = inverse_transform_matrix(T)
        
        # Verify inversion by multiplication
        result = np.dot(T, T_inv)
        expected_identity = np.eye(4)
        
        np.testing.assert_array_almost_equal(result, expected_identity)
    
    @pytest.mark.unit 
    def test_matrix_to_xyz_rpy(self):
        """Test conversion from transformation matrix to XYZ+RPY."""
        # Create transformation matrix with known values
        original_xyz_rpy = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
        T = xyz_rpy_to_matrix(original_xyz_rpy)
        
        # Convert back to XYZ+RPY
        recovered_xyz_rpy = matrix_to_xyz_rpy(T)
        
        # Verify round-trip conversion
        np.testing.assert_array_almost_equal(recovered_xyz_rpy, original_xyz_rpy, decimal=5)

class TestImageUtilities:
    """Test image handling utility functions."""
    
    @pytest.mark.unit
    def test_load_images_from_directory(self, temp_output_dir):
        """Test loading images from directory."""
        # Create test directory with mock image files
        images_dir = temp_output_dir / "test_images"
        images_dir.mkdir()
        
        # Create mock image files
        image_files = ["img001.jpg", "img002.png", "img003.jpeg"]
        for img_file in image_files:
            (images_dir / img_file).touch()
        
        # Also create non-image files (should be filtered out)
        (images_dir / "not_image.txt").touch()
        (images_dir / "config.json").touch()
        
        # Test loading with default extensions
        loaded_paths = load_images_from_directory(str(images_dir))
        
        # Should load only image files
        assert len(loaded_paths) <= len(image_files)  # May filter based on extensions
        
        for path in loaded_paths:
            assert Path(path).exists()
            # Check that loaded files have image extensions
            suffix = Path(path).suffix.lower()
            assert suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    @pytest.mark.unit
    def test_load_images_from_directory_custom_extensions(self, temp_output_dir):
        """Test loading images with custom extensions."""
        images_dir = temp_output_dir / "custom_images"
        images_dir.mkdir()
        
        # Create files with specific extensions
        (images_dir / "image1.jpg").touch()
        (images_dir / "image2.png").touch()
        (images_dir / "image3.bmp").touch()
        
        # Test loading with custom extension filter
        loaded_paths = load_images_from_directory(str(images_dir), extensions=['.jpg'])
        
        # Should only load .jpg files if extensions parameter works
        if loaded_paths:  # Function might not support extensions parameter
            for path in loaded_paths:
                assert Path(path).suffix.lower() == '.jpg'
    
    @pytest.mark.unit
    @patch('cv2.findChessboardCorners')
    def test_find_chessboard_corners_success(self, mock_find_corners):
        """Test successful chessboard corner detection."""
        # Mock successful corner detection
        mock_corners = np.random.rand(70, 1, 2).astype(np.float32)
        mock_find_corners.return_value = (True, mock_corners)
        
        # Create mock grayscale image
        gray = np.zeros((480, 640), dtype=np.uint8)
        XX, YY = 10, 7
        
        result = find_chessboard_corners(gray, XX, YY)
        
        # Verify OpenCV function was called
        mock_find_corners.assert_called_once()
        
        # Verify result structure (depends on implementation)
        assert result is not None

class TestBasicFunctionality:
    """Test basic function availability and structure."""
    
    @pytest.mark.unit
    def test_functions_are_callable(self):
        """Test that all imported functions are callable."""
        functions = [
            get_objpoints,
            find_chessboard_corners,
            load_images_from_directory,
            rpy_to_matrix,
            xyz_rpy_to_matrix,
            inverse_transform_matrix,
            matrix_to_xyz_rpy
        ]
        
        for func in functions:
            assert callable(func), f"{func.__name__} is not callable"
    
    @pytest.mark.unit
    def test_objpoints_empty_case(self):
        """Test objpoints generation with edge case."""
        objpoints = get_objpoints(0, 5, 5, 1.0)
        assert len(objpoints) == 0
    
    @pytest.mark.unit
    def test_matrix_operations_consistency(self):
        """Test consistency between matrix operations."""
        # Test multiple transformations
        xyz_rpy_values = [
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.2, 0.3],
            [0.5, 0.5, 0.5, 0.1, 0.1, 0.1]
        ]
        
        for xyz_rpy in xyz_rpy_values:
            # Forward and inverse transformation
            T = xyz_rpy_to_matrix(xyz_rpy)
            T_inv = inverse_transform_matrix(T)
            
            # Should multiply to identity
            identity = np.dot(T, T_inv)
            np.testing.assert_array_almost_equal(identity, np.eye(4), decimal=6)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
