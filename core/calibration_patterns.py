"""
Calibration Pattern Abstraction System
======================================

This module provides a comprehensive abstraction for different types of calibration 
patterns used in camera calibration, supporting both 2D planar and 3D spatial patterns.

Supported Pattern Types:
- Standard Chessboard: Traditional black and white checkerboard pattern (2D planar)
- ChArUco Board: Combination of chessboard with ArUco markers (2D planar or 3D spatial)
- Custom 3D Patterns: Patterns with known 3D coordinates in space
- Extensible architecture for adding new pattern types

The abstraction supports:
- Planar patterns (traditional flat chessboards, z=0)
- 3D spatial patterns (markers on 3D objects with known coordinates)
- Mixed calibration setups with multiple pattern types
- Custom feature detection algorithms
- Cross-version OpenCV compatibility

Author: Camera Calibration Toolkit
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
    
    def validate_dimensions(self, width: int, height: int, min_size: int = 3) -> None:
        """
        Validate pattern dimensions.
        
        Args:
            width: Pattern width
            height: Pattern height  
            min_size: Minimum allowed size for each dimension
            
        Raises:
            ValueError: If dimensions are invalid
        """
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("Width and height must be integers")
        if width < min_size or height < min_size:
            raise ValueError(f"Width and height must be at least {min_size}")
    
    def validate_physical_size(self, size: float, param_name: str = "size") -> None:
        """
        Validate physical size parameter.
        
        Args:
            size: Physical size in meters
            param_name: Name of the parameter for error messages
            
        Raises:
            ValueError: If size is invalid
        """
        if not isinstance(size, (int, float)) or size <= 0:
            raise ValueError(f"{param_name} must be a positive number")
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale if needed.
        
        Args:
            image: Input image (color or grayscale)
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def generate_planar_object_points(self, width: int, height: int, square_size: float, 
                                    start_x: float = 0.0, start_y: float = 0.0) -> np.ndarray:
        """
        Generate object points for a planar pattern.
        
        Args:
            width: Number of points along width
            height: Number of points along height
            square_size: Physical size between adjacent points
            start_x: Starting X coordinate (default 0.0)
            start_y: Starting Y coordinate (default 0.0)
            
        Returns:
            Array of 3D object points (Nx3) with Z=0
        """
        objp = np.zeros((width * height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        objp *= square_size
        objp[:, 0] += start_x
        objp[:, 1] += start_y
        return objp
        
    @abstractmethod
    def detect_corners(self, image: np.ndarray, **kwargs) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect corners/features in the given image.
        
        Args:
            image: Input image (grayscale or color)
            **kwargs: Additional parameters specific to the pattern type
            
        Returns:
            Tuple of (success, image_points, point_ids) where:
            - success: Whether detection was successful
            - image_points: Detected 2D image coordinates (Nx2 or Nx1x2)
            - point_ids: Optional IDs of detected points for 3D patterns (None for ordered patterns)
        """
        pass
    
    @abstractmethod
    def generate_object_points(self, point_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate 3D object points for the pattern.
        
        Args:
            point_ids: Optional array of point IDs for patterns with non-sequential detection
            
        Returns:
            Array of 3D object points in world coordinates (Nx3)
        """
        pass
    
    @abstractmethod
    def get_pattern_size(self) -> Union[Tuple[int, int], int]:
        """
        Get the pattern size.
        
        Returns:
            For 2D patterns: Tuple of (width, height) in number of corners/features
            For 3D patterns: Total number of detectable features
        """
        pass
    
    def is_3d_pattern(self) -> bool:
        """Check if this is a 3D spatial pattern."""
        return not self.is_planar
    
    def get_info(self) -> Dict[str, Any]:
        """Get pattern information."""
        return {
            'pattern_id': self.pattern_id,
            'name': self.name,
            'description': self.description,
            'is_planar': self.is_planar,
            'is_3d': self.is_3d_pattern(),
            'pattern_size': self.get_pattern_size()
        }
    
    def draw_corners(self, image: np.ndarray, corners: np.ndarray, 
                    point_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draw detected corners/features on the image.
        
        Args:
            image: Input image
            corners: Detected corners/features
            point_ids: Optional point IDs for labeling
            
        Returns:
            Image with corners drawn
        """
        if corners is not None:
            # Default implementation: draw circles at corner locations
            result_img = image.copy()
            corners_2d = corners.reshape(-1, 2).astype(int)
            
            for i, corner in enumerate(corners_2d):
                cv2.circle(result_img, tuple(corner), 5, (0, 255, 0), 2)
                
                # Draw point ID if available
                if point_ids is not None and i < len(point_ids):
                    cv2.putText(result_img, str(point_ids[i]), 
                              (corner[0] + 10, corner[1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            return result_img
        return image

    @abstractmethod
    def generate_pattern_image(self, pixel_per_square: int = 100, 
                             border_pixels: int = 0) -> np.ndarray:
        """
        Generate an image of the calibration pattern for display or printing.
        
        Args:
            pixel_per_square: Size of each square/unit in pixels (default: 100)
            border_pixels: Border size around the pattern in pixels (default: 0)
            
        Returns:
            Generated pattern image as numpy array (BGR format for OpenCV compatibility)
        """
        pass

    def get_pattern_description(self) -> str:
        """
        Get a brief description of the pattern for display purposes.
        
        Returns:
            Human-readable description string
        """
        info = self.get_info()
        if self.pattern_id == "standard_chessboard":
            return f"{info['width']}×{info['height']} corners, {info['square_size']*1000:.1f}mm squares"
        elif self.pattern_id == "charuco_board":
            return f"{info['width']}×{info['height']} squares, {info['square_size']*1000:.1f}mm sq, {info['marker_size']*1000:.1f}mm markers"
        else:
            return f"{info['name']} - {info.get('total_features', 'N/A')} features"


class StandardChessboard(CalibrationPattern):
    """Standard black and white chessboard pattern (2D planar)."""
    
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


class CharucoBoard(CalibrationPattern):
    """ChArUco board pattern (chessboard with ArUco markers)."""
    
    def __init__(self, width: int, height: int, square_size: float, marker_size: float, 
                 dictionary_id: int = cv2.aruco.DICT_6X6_250, is_planar: bool = True):
        """
        Initialize ChArUco board.
        
        Args:
            width: Number of squares along width
            height: Number of squares along height  
            square_size: Physical size of each square in meters
            marker_size: Physical size of ArUco markers in meters
            dictionary_id: ArUco dictionary to use
            is_planar: Whether the pattern lies in a plane (True) or has 3D structure (False)
        """
        super().__init__(
            pattern_id="charuco_board",
            name="ChArUco Board",
            description="Chessboard with ArUco markers for robust detection",
            is_planar=is_planar
        )
        
        # Use base class validation
        self.validate_dimensions(width, height, min_size=3)
        self.validate_physical_size(square_size, "square_size")
        self.validate_physical_size(marker_size, "marker_size")
        
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
            # OpenCV 4.7+ method - use CharucoDetector for newer versions
            self.detector_params = cv2.aruco.DetectorParameters()
            self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)
        except AttributeError:
            # Older OpenCV method
            self.detector_params = cv2.aruco.DetectorParameters_create()
            self.charuco_detector = None
        
    def detect_corners(self, image: np.ndarray, **kwargs) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect corners using ChArUco detection."""
        # Use base class utility to convert to grayscale
        gray = self.convert_to_grayscale(image)
            
        # Use newer CharucoDetector if available
        if self.charuco_detector is not None:
            try:
                # OpenCV 4.8+ method using CharucoDetector
                corners_charuco, ids_charuco, corners_aruco, ids_aruco = self.charuco_detector.detectBoard(gray)
                
                if corners_charuco is not None and len(corners_charuco) > 0:
                    return True, corners_charuco, ids_charuco
                else:
                    return False, None, None
            except Exception as e:
                # Fallback to older method
                pass
        
        # Fallback: Detect ArUco markers first, then interpolate ChArUco corners  
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
            # Try to interpolate ChArUco corners using older API names
            try:
                # Try different possible function names for interpolation
                interpolate_func = None
                for func_name in ['interpolateCornersCharuco', 'interpolateCorners']:
                    if hasattr(cv2.aruco, func_name):
                        interpolate_func = getattr(cv2.aruco, func_name)
                        break
                
                if interpolate_func:
                    ret, corners_charuco, ids_charuco = interpolate_func(
                        corners_aruco, ids_aruco, gray, self.charuco_board
                    )
                    if ret > 0:
                        return True, corners_charuco, ids_charuco
                        
            except Exception as e:
                # If interpolation fails, we can't use ChArUco corners
                pass
        
        return False, None, None
    
    def generate_object_points(self, point_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate 3D object points for ChArUco board."""
        # Get ChArUco board corner positions
        try:
            # Try newer OpenCV method to get all chessboard corners
            all_objp = self.charuco_board.getChessboardCorners()
        except AttributeError:
            # Fallback for older OpenCV versions
            try:
                all_objp = self.charuco_board.chessboardCorners
            except AttributeError:
                # Manual generation as fallback
                all_objp = np.zeros(((self.width - 1) * (self.height - 1), 3), np.float32)
                all_objp[:, :2] = np.mgrid[0:self.width-1, 0:self.height-1].T.reshape(-1, 2)
                all_objp *= self.square_size
        
        # For ChArUco, we need to return only the object points corresponding to detected corners
        if point_ids is not None and len(point_ids) > 0:
            # Return object points for the specific detected corner IDs
            return all_objp[point_ids.flatten()]
        else:
            # Return all possible object points
            return all_objp
    
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

    def generate_pattern_image(self, pixel_per_square: int = 100, 
                             border_pixels: int = 0) -> np.ndarray:
        """
        Generate a ChArUco board pattern image using OpenCV's proper generation methods.
        
        Args:
            pixel_per_square: Size of each square in pixels (default: 100)
            border_pixels: Border size around the pattern in pixels (default: 0)
            
        Returns:
            Generated ChArUco board image as numpy array
        """
        # Calculate image size from pattern dimensions
        total_pattern_width = self.width * pixel_per_square
        total_pattern_height = self.height * pixel_per_square
        
        image_width = total_pattern_width + 2 * border_pixels
        image_height = total_pattern_height + 2 * border_pixels
        
        # Use OpenCV's built-in ChArUco board generation
        try:
            # Method 1: Try new generateImage() method (OpenCV 4.7+)
            if hasattr(self.charuco_board, 'generateImage'):
                board_image = self.charuco_board.generateImage(
                    (total_pattern_width, total_pattern_height), 
                    marginSize=0, 
                    borderBits=1
                )
            else:
                # Method 2: Try legacy drawPlanarBoard (older OpenCV versions)
                board_image = np.ones((total_pattern_height, total_pattern_width, 3), dtype=np.uint8) * 255
                cv2.aruco.drawPlanarBoard(
                    self.charuco_board, 
                    (total_pattern_width, total_pattern_height), 
                    board_image, 
                    marginSize=0, 
                    borderBits=1
                )
                
        except (AttributeError, TypeError) as e:
            # Method 3: Manual fallback for very old OpenCV versions
            print(f"Warning: Using manual ChArUco generation due to: {e}")
            return self._generate_manual_charuco_pattern(pixel_per_square, border_pixels)
        
        # Convert to 3-channel BGR if needed
        if len(board_image.shape) == 2:
            board_image = cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)
        
        # Add border if requested
        if border_pixels > 0:
            # Create final image with border
            final_image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
            
            # Place the board image in the center
            start_x = border_pixels
            start_y = border_pixels
            end_x = start_x + total_pattern_width
            end_y = start_y + total_pattern_height
            
            final_image[start_y:end_y, start_x:end_x] = board_image
            
            return final_image
        else:
            return board_image
    
    def _generate_manual_charuco_pattern(self, pixel_per_square: int = 100, 
                                       border_pixels: int = 0) -> np.ndarray:
        """
        Manual ChArUco pattern generation as fallback for older OpenCV versions.
        This is the old implementation kept as a backup.
        """
        # Calculate image size from pattern dimensions
        total_pattern_width = self.width * pixel_per_square
        total_pattern_height = self.height * pixel_per_square
        
        image_width = total_pattern_width + 2 * border_pixels
        image_height = total_pattern_height + 2 * border_pixels
        
        # Create blank white image
        image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
        
        # Calculate starting position
        start_x = border_pixels
        start_y = border_pixels
        
        # Calculate marker size in pixels
        marker_ratio = self.marker_size / self.square_size
        marker_size_px = pixel_per_square * marker_ratio
        
        marker_id = 0
        
        # Draw ChArUco pattern manually
        for row in range(self.height):
            for col in range(self.width):
                # Calculate square position
                x1 = int(start_x + col * pixel_per_square)
                y1 = int(start_y + row * pixel_per_square)
                x2 = int(start_x + (col + 1) * pixel_per_square)
                y2 = int(start_y + (row + 1) * pixel_per_square)
                
                # Determine square color (ChArUco pattern)
                is_black = (row + col) % 2 == 0
                
                if is_black:
                    # Black square with ArUco marker
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
                    
                    # Draw simplified ArUco marker representation
                    marker_x1 = int(x1 + (pixel_per_square - marker_size_px) / 2)
                    marker_y1 = int(y1 + (pixel_per_square - marker_size_px) / 2)
                    marker_x2 = int(marker_x1 + marker_size_px)
                    marker_y2 = int(marker_y1 + marker_size_px)
                    
                    # White marker background
                    cv2.rectangle(image, (marker_x1, marker_y1), (marker_x2, marker_y2), (255, 255, 255), -1)
                    
                    # Black border around marker
                    cv2.rectangle(image, (marker_x1, marker_y1), (marker_x2, marker_y2), (0, 0, 0), 2)
                    
                    marker_id += 1
                else:
                    # White square
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
                
                # Add square border for clarity
                cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 128), 1)
        
        # Add outer border
        border_thickness = max(2, int(pixel_per_square / 30))
        cv2.rectangle(image, 
                     (int(start_x), int(start_y)), 
                     (int(start_x + total_pattern_width), int(start_y + total_pattern_height)), 
                     (0, 0, 0), border_thickness)
        
        return image
    
    @classmethod
    def get_configuration_schema(cls):
        """
        Get the configuration schema for this pattern type.
        
        Returns:
            Dict containing the pattern configuration schema
        """
        return {
            "name": "ChArUco Board",
            "description": "Chessboard pattern with ArUco markers for robust detection",
            "icon": "🎯",
            "parameters": [
                {
                    "name": "width",
                    "label": "Squares (Width)",
                    "type": "integer", 
                    "default": 8,
                    "min": 3,
                    "max": 15,
                    "description": "Number of squares along width"
                },
                {
                    "name": "height",
                    "label": "Squares (Height)", 
                    "type": "integer",
                    "default": 6,
                    "min": 3,
                    "max": 15,
                    "description": "Number of squares along height"
                },
                {
                    "name": "square_size",
                    "label": "Square Size (meters)",
                    "type": "float",
                    "default": 0.040,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.001,
                    "description": "Physical size of each square in meters"
                },
                {
                    "name": "marker_size",
                    "label": "Marker Size (meters)",
                    "type": "float",
                    "default": 0.020,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.001,
                    "description": "Physical size of ArUco markers in meters"
                },
                {
                    "name": "dictionary_id",
                    "label": "ArUco Dictionary",
                    "type": "integer",
                    "input_type": "select",
                    "default": cv2.aruco.DICT_6X6_250,
                    "options": [
                        {"value": cv2.aruco.DICT_4X4_50, "label": "DICT_4X4_50"},
                        {"value": cv2.aruco.DICT_4X4_100, "label": "DICT_4X4_100"},
                        {"value": cv2.aruco.DICT_4X4_250, "label": "DICT_4X4_250"},
                        {"value": cv2.aruco.DICT_4X4_1000, "label": "DICT_4X4_1000"},
                        {"value": cv2.aruco.DICT_5X5_50, "label": "DICT_5X5_50"},
                        {"value": cv2.aruco.DICT_5X5_100, "label": "DICT_5X5_100"},
                        {"value": cv2.aruco.DICT_5X5_250, "label": "DICT_5X5_250"},
                        {"value": cv2.aruco.DICT_5X5_1000, "label": "DICT_5X5_1000"},
                        {"value": cv2.aruco.DICT_6X6_50, "label": "DICT_6X6_50"},
                        {"value": cv2.aruco.DICT_6X6_100, "label": "DICT_6X6_100"},
                        {"value": cv2.aruco.DICT_6X6_250, "label": "DICT_6X6_250"},
                        {"value": cv2.aruco.DICT_6X6_1000, "label": "DICT_6X6_1000"}
                    ],
                    "description": "ArUco marker dictionary to use"
                }
            ]
        }


class Custom3DPattern(CalibrationPattern):
    """Custom 3D calibration pattern with known 3D coordinates."""
    
    def __init__(self, pattern_id: str, name: str, object_points_3d: np.ndarray, 
                 feature_detector: callable = None):
        """
        Initialize custom 3D pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern
            name: Human-readable name
            object_points_3d: Known 3D coordinates of pattern features (Nx3)
            feature_detector: Function to detect 2D features in images
        """
        super().__init__(
            pattern_id=pattern_id,
            name=name,
            description="Custom 3D calibration pattern with known coordinates",
            is_planar=False
        )
        self.object_points_3d = np.array(object_points_3d)
        self.feature_detector = feature_detector
        
        if self.object_points_3d.shape[1] != 3:
            raise ValueError("Object points must be Nx3 array (x, y, z coordinates)")
    
    def detect_corners(self, image: np.ndarray, **kwargs) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect features using custom detector."""
        if self.feature_detector is None:
            raise NotImplementedError("Feature detector must be provided for custom 3D patterns")
        
        try:
            success, image_points, point_ids = self.feature_detector(image, **kwargs)
            return success, image_points, point_ids
        except Exception as e:
            print(f"Feature detection failed: {e}")
            return False, None, None
    
    def generate_object_points(self, point_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate 3D object points for detected features."""
        if point_ids is not None:
            # Return object points for specific detected features
            point_ids_flat = point_ids.flatten()
            if np.max(point_ids_flat) >= len(self.object_points_3d):
                raise ValueError(f"Point ID {np.max(point_ids_flat)} exceeds available object points")
            return self.object_points_3d[point_ids_flat]
        else:
            # Return all object points
            return self.object_points_3d
    
    def get_pattern_size(self) -> int:
        """Get total number of features in the pattern."""
        return len(self.object_points_3d)
    
    def get_info(self) -> Dict[str, Any]:
        """Get pattern information."""
        info = super().get_info()
        info.update({
            'total_features': len(self.object_points_3d),
            'coordinate_range': {
                'x_range': [float(self.object_points_3d[:, 0].min()), float(self.object_points_3d[:, 0].max())],
                'y_range': [float(self.object_points_3d[:, 1].min()), float(self.object_points_3d[:, 1].max())],
                'z_range': [float(self.object_points_3d[:, 2].min()), float(self.object_points_3d[:, 2].max())]
            }
        })
        return info

    def generate_pattern_image(self, pixel_per_square: int = 100, 
                             border_pixels: int = 0) -> np.ndarray:
        """
        Generate a visualization of the 3D pattern projected to 2D.
        
        Args:
            pixel_per_square: Not used for 3D patterns (for compatibility)
            border_pixels: Border size around the pattern in pixels (default: 0)
            
        Returns:
            Generated pattern visualization as numpy array
        """
        # Default size for 3D pattern visualization
        image_width = 400 + 2 * border_pixels
        image_height = 300 + 2 * border_pixels
        
        # Create blank white image
        image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
        
        # Get 2D projection of 3D points (ignoring Z for simple visualization)
        points_2d = self.object_points_3d[:, :2]  # Take only X and Y
        
        if len(points_2d) == 0:
            return image
        
        # Normalize points to fit in image with border
        x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
        y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
        
        # Scale points to fit in image with border
        if x_max != x_min and y_max != y_min:
            available_width = image_width - 2 * border_pixels
            available_height = image_height - 2 * border_pixels
            
            scale_x = available_width / (x_max - x_min)
            scale_y = available_height / (y_max - y_min)
            scale = min(scale_x, scale_y)
            
            # Transform points to image coordinates
            points_img = np.zeros_like(points_2d)
            points_img[:, 0] = (points_2d[:, 0] - x_min) * scale + border_pixels
            points_img[:, 1] = (points_2d[:, 1] - y_min) * scale + border_pixels
            
            # Draw points as circles
            radius = max(3, int(scale * 0.01))
            for i, point in enumerate(points_img):
                center = (int(point[0]), int(point[1]))
                cv2.circle(image, center, radius, (0, 100, 200), -1)
                
                # Add point index
                cv2.putText(image, str(i), (center[0] + radius + 2, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Add title
        text = f"3D Pattern ({len(points_2d)} points)"
        cv2.putText(image, text, (border_pixels + 10, border_pixels + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        return image


class CalibrationPatternManager:
    """Manager for different calibration pattern types."""
    
    def __init__(self):
        """Initialize calibration pattern manager."""
        self.patterns = {}
        self.register_default_patterns()
    
    def register_default_patterns(self):
        """Register default pattern types."""
        # Standard chessboard patterns
        self.register_pattern_type("standard", StandardChessboard)
        self.register_pattern_type("charuco", CharucoBoard)
        self.register_pattern_type("custom3d", Custom3DPattern)
    
    def register_pattern_type(self, pattern_type: str, pattern_class: type):
        """Register a new pattern type."""
        self.patterns[pattern_type] = pattern_class
    
    def create_pattern(self, pattern_type: str, **kwargs) -> CalibrationPattern:
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
def create_chessboard_pattern(pattern_type: str, **kwargs) -> CalibrationPattern:
    """
    Factory function to create calibration patterns.
    
    Args:
        pattern_type: Type of pattern ('standard', 'charuco', 'custom3d')
        **kwargs: Pattern-specific parameters
        
    Returns:
        CalibrationPattern instance
    """
    manager = CalibrationPatternManager()
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


def get_common_pattern(pattern_name: str) -> CalibrationPattern:
    """
    Get a pre-configured common pattern.
    
    Args:
        pattern_name: Name of the common pattern
        
    Returns:
        CalibrationPattern instance
    """
    if pattern_name not in COMMON_PATTERNS:
        raise ValueError(f"Unknown common pattern: {pattern_name}. "
                        f"Available: {list(COMMON_PATTERNS.keys())}")
    
    config = COMMON_PATTERNS[pattern_name]
    pattern_type = config.pop("type")
    config.pop("description", None)  # Remove description from parameters
    
    return create_chessboard_pattern(pattern_type, **config)


def get_pattern_type_configurations():
    """
    Get available pattern types and their configuration parameters.
    Collects configuration schemas from all pattern classes.
    
    Returns:
        Dict containing pattern types and their configuration schemas
    """
    # Registry of available pattern classes
    # To add a new pattern type, simply add it to this dictionary
    pattern_classes = {
        "standard": StandardChessboard,
        "charuco": CharucoBoard,
        # Future pattern types can be added here:
        # "circles": CircleGrid,
        # "apriltag": AprilTagBoard,
        # "custom3d": Custom3DPattern,
    }
    
    # Collect configurations from all pattern classes
    pattern_configs = {}
    for pattern_type, pattern_class in pattern_classes.items():
        if hasattr(pattern_class, 'get_configuration_schema'):
            pattern_configs[pattern_type] = pattern_class.get_configuration_schema()
        else:
            # Fallback for classes without configuration schema
            pattern_configs[pattern_type] = {
                "name": pattern_class.__name__,
                "description": pattern_class.__doc__ or "No description available",
                "icon": "❓",
                "parameters": []
            }
    
    return pattern_configs
