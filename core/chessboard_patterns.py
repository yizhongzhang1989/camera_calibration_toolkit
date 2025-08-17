"""
Chessboard Pattern Abstraction
=============================

This module provides an abstraction for different types of chessboard patterns
used in camera calibration, including common chessboards and ChArUco boards.

Supported Pattern Types:
- Standard Chessboard: Traditional black and white checkerboard pattern
- ChArUco Board: Combination of chessboard with ArUco markers for improved detection

Author: Camera Calibration Toolkit
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any
import json


class ChessboardPattern(ABC):
    """Abstract base class for chessboard patterns used in camera calibration."""
    
    def __init__(self, pattern_id: str, name: str, description: str):
        """
        Initialize chessboard pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern type
            name: Human-readable name of the pattern
            description: Description of the pattern
        """
        self.pattern_id = pattern_id
        self.name = name
        self.description = description
        self.detected_corners = []
        self.object_points = []
        
    @abstractmethod
    def detect_corners(self, image: np.ndarray, **kwargs) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect corners in the given image.
        
        Args:
            image: Input image (grayscale or color)
            **kwargs: Additional parameters specific to the pattern type
            
        Returns:
            Tuple of (success, corners) where corners is None if detection failed
        """
        pass
    
    @abstractmethod
    def generate_object_points(self) -> np.ndarray:
        """
        Generate 3D object points for the pattern.
        
        Returns:
            Array of 3D object points in world coordinates
        """
        pass
    
    @abstractmethod
    def get_pattern_size(self) -> Tuple[int, int]:
        """
        Get the pattern size (number of internal corners).
        
        Returns:
            Tuple of (width, height) in number of corners
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get pattern information."""
        return {
            'pattern_id': self.pattern_id,
            'name': self.name,
            'description': self.description,
            'pattern_size': self.get_pattern_size()
        }
    
    def draw_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Draw detected corners on the image.
        
        Args:
            image: Input image
            corners: Detected corners
            
        Returns:
            Image with corners drawn
        """
        if corners is not None:
            pattern_size = self.get_pattern_size()
            return cv2.drawChessboardCorners(image, pattern_size, corners, True)
        return image


class StandardChessboard(ChessboardPattern):
    """Standard black and white chessboard pattern."""
    
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
            description="Traditional black and white checkerboard pattern"
        )
        self.width = width
        self.height = height
        self.square_size = square_size
        
    def detect_corners(self, image: np.ndarray, **kwargs) -> Tuple[bool, Optional[np.ndarray]]:
        """Detect corners using standard chessboard detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detection parameters
        flags = kwargs.get('flags', cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        # Find chessboard corners
        pattern_size = (self.width, self.height)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
        return ret, corners if ret else None
    
    def generate_object_points(self) -> np.ndarray:
        """Generate 3D object points for standard chessboard."""
        # Create 3D points (z=0 for planar pattern)
        objp = np.zeros((self.width * self.height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.width, 0:self.height].T.reshape(-1, 2)
        objp *= self.square_size
        
        return objp
    
    def get_pattern_size(self) -> Tuple[int, int]:
        """Get pattern size."""
        return (self.width, self.height)
    
    def get_info(self) -> Dict[str, Any]:
        """Get pattern information."""
        info = super().get_info()
        info.update({
            'width': self.width,
            'height': self.height,
            'square_size': self.square_size,
            'total_corners': self.width * self.height
        })
        return info


class CharucoBoard(ChessboardPattern):
    """ChArUco board pattern (chessboard with ArUco markers)."""
    
    def __init__(self, width: int, height: int, square_size: float, marker_size: float, 
                 dictionary_id: int = cv2.aruco.DICT_6X6_250):
        """
        Initialize ChArUco board.
        
        Args:
            width: Number of squares along width
            height: Number of squares along height  
            square_size: Physical size of each square in meters
            marker_size: Physical size of ArUco markers in meters
            dictionary_id: ArUco dictionary to use
        """
        super().__init__(
            pattern_id="charuco_board",
            name="ChArUco Board",
            description="Chessboard with ArUco markers for robust detection"
        )
        self.width = width
        self.height = height
        self.square_size = square_size
        self.marker_size = marker_size
        self.dictionary_id = dictionary_id
        
        # Create ArUco dictionary and ChArUco board
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        
        # Try different ChArUco board creation methods for different OpenCV versions
        try:
            # OpenCV 4.7+ method
            self.charuco_board = cv2.aruco.CharucoBoard(
                (width, height), square_size, marker_size, self.aruco_dict
            )
        except (AttributeError, TypeError):
            try:
                # Older OpenCV method
                self.charuco_board = cv2.aruco.CharucoBoard_create(
                    width, height, square_size, marker_size, self.aruco_dict
                )
            except AttributeError:
                raise ValueError("ChArUco boards are not supported in this OpenCV version")
        
        # Detector parameters
        try:
            # OpenCV 4.7+ method
            self.detector_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            # Older OpenCV method
            self.detector_params = cv2.aruco.DetectorParameters_create()
        
    def detect_corners(self, image: np.ndarray, **kwargs) -> Tuple[bool, Optional[np.ndarray]]:
        """Detect corners using ChArUco detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect ArUco markers first
        try:
            # OpenCV 4.7+ method
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
            corners_aruco, ids_aruco, _ = detector.detectMarkers(gray)
        except AttributeError:
            # Older OpenCV method
            corners_aruco, ids_aruco, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.detector_params
            )
        
        if len(corners_aruco) > 0:
            # Interpolate ChArUco corners
            try:
                # Try newer OpenCV method
                ret, corners_charuco, ids_charuco = cv2.aruco.interpolateCornersCharuco(
                    corners_aruco, ids_aruco, gray, self.charuco_board
                )
            except Exception:
                # Fallback for older versions or different parameters
                try:
                    ret, corners_charuco, ids_charuco = cv2.aruco.interpolateCornersCharuco(
                        corners_aruco, ids_aruco, gray, self.charuco_board.getBoard()
                    )
                except Exception:
                    return False, None
            
            if ret > 0:
                return True, corners_charuco
        
        return False, None
    
    def generate_object_points(self) -> np.ndarray:
        """Generate 3D object points for ChArUco board."""
        # Get ChArUco board corner positions
        try:
            # Try newer OpenCV method
            objp = self.charuco_board.getChessboardCorners()
        except AttributeError:
            # Fallback for older OpenCV versions
            try:
                objp = self.charuco_board.chessboardCorners
            except AttributeError:
                # Manual generation as fallback
                objp = np.zeros(((self.width - 1) * (self.height - 1), 3), np.float32)
                objp[:, :2] = np.mgrid[0:self.width-1, 0:self.height-1].T.reshape(-1, 2)
                objp *= self.square_size
        
        return objp
    
    def get_pattern_size(self) -> Tuple[int, int]:
        """Get pattern size (internal corners)."""
        return (self.width - 1, self.height - 1)
    
    def get_info(self) -> Dict[str, Any]:
        """Get pattern information."""
        info = super().get_info()
        info.update({
            'width': self.width,
            'height': self.height,
            'square_size': self.square_size,
            'marker_size': self.marker_size,
            'dictionary_id': self.dictionary_id,
            'total_corners': (self.width - 1) * (self.height - 1)
        })
        return info
    
    def draw_corners(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Draw detected ChArUco corners."""
        if corners is not None:
            return cv2.aruco.drawDetectedCornersCharuco(image, corners)
        return image


class ChessboardManager:
    """Manager for different chessboard patterns."""
    
    def __init__(self):
        """Initialize chessboard manager."""
        self.patterns = {}
        self.register_default_patterns()
    
    def register_default_patterns(self):
        """Register default pattern types."""
        # Standard chessboard patterns
        self.register_pattern_type("standard", StandardChessboard)
        self.register_pattern_type("charuco", CharucoBoard)
    
    def register_pattern_type(self, pattern_type: str, pattern_class: type):
        """Register a new pattern type."""
        self.patterns[pattern_type] = pattern_class
    
    def create_pattern(self, pattern_type: str, **kwargs) -> ChessboardPattern:
        """
        Create a chessboard pattern instance.
        
        Args:
            pattern_type: Type of pattern ('standard' or 'charuco')
            **kwargs: Pattern-specific parameters
            
        Returns:
            ChessboardPattern instance
            
        Raises:
            ValueError: If pattern type is not supported
        """
        if pattern_type not in self.patterns:
            raise ValueError(f"Unsupported pattern type: {pattern_type}. "
                           f"Available types: {list(self.patterns.keys())}")
        
        pattern_class = self.patterns[pattern_type]
        
        if pattern_type == "standard":
            required_params = ['width', 'height', 'square_size']
            for param in required_params:
                if param not in kwargs:
                    raise ValueError(f"Missing required parameter '{param}' for standard chessboard")
            
            return pattern_class(
                width=kwargs['width'],
                height=kwargs['height'],
                square_size=kwargs['square_size']
            )
            
        elif pattern_type == "charuco":
            required_params = ['width', 'height', 'square_size', 'marker_size']
            for param in required_params:
                if param not in kwargs:
                    raise ValueError(f"Missing required parameter '{param}' for ChArUco board")
            
            return pattern_class(
                width=kwargs['width'],
                height=kwargs['height'],
                square_size=kwargs['square_size'],
                marker_size=kwargs['marker_size'],
                dictionary_id=kwargs.get('dictionary_id', cv2.aruco.DICT_6X6_250)
            )
        
        else:
            # For custom pattern types
            return pattern_class(**kwargs)
    
    def get_available_patterns(self) -> List[str]:
        """Get list of available pattern types."""
        return list(self.patterns.keys())
    
    def get_pattern_info(self, pattern_type: str) -> Dict[str, Any]:
        """Get information about a pattern type."""
        if pattern_type not in self.patterns:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        # Create a temporary instance to get info (with dummy parameters)
        if pattern_type == "standard":
            temp_pattern = self.patterns[pattern_type](9, 6, 0.025)
        elif pattern_type == "charuco":
            temp_pattern = self.patterns[pattern_type](5, 7, 0.04, 0.02)
        else:
            return {"pattern_id": pattern_type, "name": "Custom Pattern"}
        
        return temp_pattern.get_info()


# Factory function for easy access
def create_chessboard_pattern(pattern_type: str, **kwargs) -> ChessboardPattern:
    """
    Factory function to create chessboard patterns.
    
    Args:
        pattern_type: Type of pattern ('standard' or 'charuco')
        **kwargs: Pattern-specific parameters
        
    Returns:
        ChessboardPattern instance
    """
    manager = ChessboardManager()
    return manager.create_pattern(pattern_type, **kwargs)


# Common pattern configurations
COMMON_PATTERNS = {
    "standard_9x6": {
        "type": "standard",
        "width": 9,
        "height": 6, 
        "square_size": 0.025,
        "description": "Standard 9x6 chessboard with 25mm squares"
    },
    "standard_11x8": {
        "type": "standard", 
        "width": 11,
        "height": 8,
        "square_size": 0.020,
        "description": "Standard 11x8 chessboard with 20mm squares"
    },
    "charuco_5x7": {
        "type": "charuco",
        "width": 5,
        "height": 7,
        "square_size": 0.040,
        "marker_size": 0.020,
        "description": "ChArUco 5x7 board with 40mm squares and 20mm markers"
    }
}


def get_common_pattern(pattern_name: str) -> ChessboardPattern:
    """
    Get a pre-configured common pattern.
    
    Args:
        pattern_name: Name of the common pattern
        
    Returns:
        ChessboardPattern instance
    """
    if pattern_name not in COMMON_PATTERNS:
        raise ValueError(f"Unknown common pattern: {pattern_name}. "
                        f"Available: {list(COMMON_PATTERNS.keys())}")
    
    config = COMMON_PATTERNS[pattern_name]
    pattern_type = config.pop("type")
    config.pop("description", None)  # Remove description from parameters
    
    return create_chessboard_pattern(pattern_type, **config)
