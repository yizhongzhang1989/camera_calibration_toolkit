"""
Standard Chessboard Pattern
==========================

Traditional black and white checkerboard pattern for camera calibration.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from .base import CalibrationPattern


class StandardChessboard(CalibrationPattern):
    """Standard black and white chessboard pattern (2D planar)."""
    
    # Pattern information for auto-discovery
    PATTERN_INFO = {
        'id': 'standard_chessboard',
        'name': 'Standard Chessboard',
        'icon': '🏁',
        'category': 'chessboard'
    }
    
    def __init__(self, width: int, height: int, square_size: float):
        """
        Initialize standard chessboard.
        
        Args:
            width: Number of internal corners along width (columns - 1)
            height: Number of internal corners along height (rows - 1)
            square_size: Physical size of each square in meters
        """
        super().__init__(
            pattern_id="standard_chessboard",
            name="Standard Chessboard",
            description="Traditional black and white checkerboard pattern",
            is_planar=True
        )
        
        # Use base class validation
        self.validate_dimensions(width, height, min_size=2)
        self.validate_physical_size(square_size, "square_size")
        
        self.width = width
        self.height = height
        self.square_size = square_size
    
    @classmethod
    def get_configuration_schema(cls):
        """
        Get the configuration schema for this pattern type.
        
        Returns:
            Dict containing the pattern configuration schema
        """
        return {
            "name": "Standard Chessboard",
            "description": "Traditional black and white checkerboard pattern",
            "icon": "🏁",
            "parameters": [
                {
                    "name": "width",
                    "label": "Corners (Width)",
                    "type": "integer",
                    "default": 11,
                    "min": 3,
                    "max": 20,
                    "description": "Number of internal corners along width"
                },
                {
                    "name": "height", 
                    "label": "Corners (Height)",
                    "type": "integer",
                    "default": 8,
                    "min": 3,
                    "max": 20,
                    "description": "Number of internal corners along height"
                },
                {
                    "name": "square_size",
                    "label": "Square Size (meters)",
                    "type": "float",
                    "default": 0.025,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.001,
                    "description": "Physical size of each square in meters"
                }
            ]
        }
        
    def detect_corners(self, image: np.ndarray, **kwargs) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect corners using standard chessboard detection."""
        # Use base class utility to convert to grayscale
        gray = self.convert_to_grayscale(image)
            
        # Detection parameters
        flags = kwargs.get('flags', cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        # Find chessboard corners
        pattern_size = (self.width, self.height)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
        # Standard chessboard has ordered corners, no IDs needed
        return ret, corners if ret else None, None
    
    def generate_object_points(self, point_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate 3D object points for standard chessboard (planar, z=0)."""
        # Use base class utility for planar object point generation
        return self.generate_planar_object_points(self.width, self.height, self.square_size)
    
    def get_pattern_size(self) -> Tuple[int, int]:
        """Get pattern size."""
        return (self.width, self.height)
    
    def draw_corners(self, image: np.ndarray, corners: np.ndarray, 
                    point_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw chessboard corners using OpenCV's specialized function."""
        if corners is not None:
            pattern_size = self.get_pattern_size()
            return cv2.drawChessboardCorners(image, pattern_size, corners, True)
        return image

    def generate_pattern_image(self, pixel_per_square: int = 100, 
                             border_pixels: int = 0) -> np.ndarray:
        """
        Generate a chessboard pattern image for display or printing.
        
        Args:
            pixel_per_square: Size of each square in pixels (default: 100)
            border_pixels: Border size around the pattern in pixels (default: 0)
            
        Returns:
            Generated chessboard image as numpy array
        """
        # Calculate number of squares (corners + 1)
        squares_x = self.width + 1
        squares_y = self.height + 1
        
        # Calculate image size from pattern dimensions
        total_pattern_width = squares_x * pixel_per_square
        total_pattern_height = squares_y * pixel_per_square
        
        image_width = total_pattern_width + 2 * border_pixels
        image_height = total_pattern_height + 2 * border_pixels
        
        # Create blank white image
        image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
        
        # Calculate starting position to center the chessboard
        start_x = border_pixels
        start_y = border_pixels
        
        # Draw chessboard squares
        for row in range(squares_y):
            for col in range(squares_x):
                # Calculate square position
                x1 = int(start_x + col * pixel_per_square)
                y1 = int(start_y + row * pixel_per_square)
                x2 = int(start_x + (col + 1) * pixel_per_square)
                y2 = int(start_y + (row + 1) * pixel_per_square)
                
                # Determine if this square should be black
                is_black = (row + col) % 2 == 0
                
                if is_black:
                    # Fill black square
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        return image
    
    def _get_parameters_dict(self) -> Dict[str, Any]:
        """Get pattern parameters for JSON serialization."""
        return {
            'width': self.width,
            'height': self.height,
            'square_size': self.square_size,
            'total_corners': self.width * self.height
        }
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'StandardChessboard':
        """Create StandardChessboard from JSON data."""
        params = json_data.get('parameters', {})
        return cls(
            width=params.get('width', 11),
            height=params.get('height', 8),
            square_size=params.get('square_size', 0.025)
        )
