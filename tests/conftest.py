"""
pytest configuration file for camera calibration toolkit
"""
import pytest
import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
pytest_plugins = []

@pytest.fixture(scope="session")
def project_root_dir():
    """Get the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def test_data_dir(project_root_dir):
    """Get the test data directory."""
    return project_root_dir / "tests" / "fixtures"

@pytest.fixture(scope="session")
def sample_data_dir(project_root_dir):
    """Get the sample data directory."""
    return project_root_dir / "sample_data"

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir

@pytest.fixture
def sample_chessboard_config():
    """Standard chessboard configuration for testing."""
    return {
        "pattern_id": "standard_chessboard",
        "name": "Standard Chessboard",
        "description": "Traditional black and white checkerboard pattern",
        "is_planar": True,
        "parameters": {
            "width": 9,
            "height": 6,
            "square_size": 0.025
        }
    }

@pytest.fixture
def sample_charuco_config():
    """ChArUco board configuration for testing."""
    return {
        "pattern_id": "charuco_board",
        "name": "ChArUco Board",
        "description": "Chessboard pattern with ArUco markers for robust detection",
        "is_planar": True,
        "parameters": {
            "width": 8,
            "height": 6,
            "square_size": 0.04,
            "marker_size": 0.02,
            "dictionary_id": 10
        }
    }

@pytest.fixture
def sample_gridboard_config():
    """ArUco Grid Board configuration for testing."""
    return {
        "pattern_id": "grid_board",
        "name": "ArUco Grid Board", 
        "description": "Grid of ArUco markers for robust camera calibration detection",
        "is_planar": True,
        "parameters": {
            "markers_x": 5,
            "markers_y": 7,
            "marker_size": 0.04,
            "marker_separation": 0.01,
            "dictionary_id": 10
        }
    }

@pytest.fixture
def mock_camera_matrix():
    """Sample camera matrix for testing."""
    return np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

@pytest.fixture
def mock_distortion_coefficients():
    """Sample distortion coefficients for testing."""
    return np.array([-0.2, 0.1, 0.001, 0.001, -0.05], dtype=np.float64)

@pytest.fixture
def synthetic_chessboard_image():
    """Generate a synthetic chessboard image for testing."""
    # Create a simple synthetic chessboard
    img_size = (640, 480)
    square_size_pixels = 40
    
    # Create chessboard pattern
    rows, cols = 6, 9
    img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                y1 = i * square_size_pixels + 50
                y2 = (i + 1) * square_size_pixels + 50
                x1 = j * square_size_pixels + 50  
                x2 = (j + 1) * square_size_pixels + 50
                
                if y2 < img_size[1] and x2 < img_size[0]:
                    img[y1:y2, x1:x2] = 255
    
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Configure test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and organize tests."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark tests that require OpenCV
        if any(module in str(item.fspath) for module in ["calibration", "pattern"]):
            item.add_marker(pytest.mark.opencv)

# Test markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "opencv: marks tests that require OpenCV"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
