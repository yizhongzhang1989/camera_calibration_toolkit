"""
Base Classes for Calibration Pattern System
==========================================

This module contains the abstract base class and common utilities
for all calibration patterns in the modular pattern system.
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any, Union
import json


class CalibrationPattern(ABC):
    """Abstract base class for calibration patterns used in camera calibration."""
    
    def __init__(self, pattern_id: str, name: str, description: str, is_planar: bool = True):
        """
        Initialize calibration pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern type
            name: Human-readable name of the pattern
            description: Description of the pattern
            is_planar: Whether the pattern lies in a single plane (z=0) or has 3D structure
        """
        self.pattern_id = pattern_id
        self.name = name
        self.description = description
        self.is_planar = is_planar
        self.detected_corners = []
        self.object_points = []
    
    @classmethod
    @abstractmethod
    def get_configuration_schema(cls) -> Dict[str, Any]:
        """
        Get the configuration schema for this pattern type.
        
        Returns:
            Dict containing the pattern configuration schema with UI information
        """
        pass
    
    @abstractmethod
    def detect_corners(self, image: np.ndarray, **kwargs) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect pattern features in an image.
        
        Args:
            image: Input image (BGR or grayscale)
            **kwargs: Additional detection parameters
            
        Returns:
            Tuple of (success, image_points, point_ids)
            - success: Whether detection was successful
            - image_points: Detected 2D points in image coordinates
            - point_ids: IDs/indices of detected points (None for ordered patterns)
        """
        pass
    
    @abstractmethod
    def generate_object_points(self, point_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate corresponding 3D object points for detected image points.
        
        Args:
            point_ids: IDs of detected points (for sparse patterns)
            
        Returns:
            3D object points corresponding to detected image points
        """
        pass
    
    @abstractmethod
    def draw_corners(self, image: np.ndarray, corners: np.ndarray, 
                    point_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draw detected corners/features on the image for visualization.
        
        Args:
            image: Input image
            corners: Detected corner points
            point_ids: IDs of detected points (optional)
            
        Returns:
            Image with drawn corners
        """
        pass
    
    @abstractmethod
    def generate_pattern_image(self, **kwargs) -> np.ndarray:
        """
        Generate a visual representation of the calibration pattern.
        
        Args:
            **kwargs: Pattern-specific generation parameters
            
        Returns:
            Generated pattern image
        """
        pass
    
    # Common utility methods that can be used by all pattern types
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def validate_dimensions(self, width: int, height: int, min_size: int = 2):
        """Validate pattern dimensions."""
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("Width and height must be integers")
        if width < min_size or height < min_size:
            raise ValueError(f"Width and height must be at least {min_size}")
    
    def validate_physical_size(self, size: float, parameter_name: str):
        """Validate physical size parameter."""
        if not isinstance(size, (int, float)) or size <= 0:
            raise ValueError(f"{parameter_name} must be a positive number")
    
    def generate_planar_object_points(self, width: int, height: int, square_size: float) -> np.ndarray:
        """Generate 3D object points for planar patterns (z=0)."""
        objp = np.zeros((width * height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        objp *= square_size
        return objp
    
    # JSON serialization methods
    
    def to_json(self) -> Dict[str, Any]:
        """Convert pattern to JSON representation."""
        return {
            'pattern_id': self.pattern_id,
            'name': self.name,
            'description': self.description,
            'is_planar': self.is_planar,
            'parameters': self._get_parameters_dict()
        }
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]):
        """Create pattern instance from JSON data."""
        # This should be implemented by each concrete pattern class
        raise NotImplementedError("Subclasses must implement from_json method")
    
    def _get_parameters_dict(self) -> Dict[str, Any]:
        """Get pattern-specific parameters as dictionary (to be overridden)."""
        return {}
    
    # Information methods
    
    def get_info(self) -> Dict[str, Any]:
        """Get pattern information dictionary."""
        info = {
            'pattern_id': self.pattern_id,
            'name': self.name,
            'description': self.description,
            'is_planar': self.is_planar
        }
        info.update(self._get_parameters_dict())
        return info
    
    def get_display_name(self) -> str:
        """Get display name for the pattern."""
        info = self._get_parameters_dict()
        if self.pattern_id == "standard_chessboard":
            return f"{info['width']}×{info['height']} corners, {info['square_size']*1000:.1f}mm squares"
        elif self.pattern_id == "charuco_board":
            return f"{info['width']}×{info['height']} squares, {info['square_size']*1000:.1f}mm sq, {info['marker_size']*1000:.1f}mm markers"
        elif self.pattern_id == "grid_board":
            return f"{info['markers_x']}×{info['markers_y']} ArUco markers, {info['marker_size']*1000:.1f}mm size, {info['marker_separation']*1000:.1f}mm sep"
        else:
            return f"{info['name']} - {info.get('total_features', 'N/A')} features"
