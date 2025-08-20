"""
ArUco Grid Board Pattern
========================

ArUco grid board pattern for camera calibration using ArUco markers
arranged in a regular grid pattern.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from .base import CalibrationPattern


class GridBoard(CalibrationPattern):
    """ArUco Grid Board pattern for camera calibration."""
    
    # Pattern information for auto-discovery
    PATTERN_INFO = {
        'id': 'grid_board',
        'name': 'ArUco Grid Board',
        'icon': '🎲',
        'category': 'aruco',
        'metadata': {
            'validation_rules': {
                'parameter_relationships': [
                    {
                        'param1': 'marker_size',
                        'param2': 'marker_separation', 
                        'constraint': 'greater_than',
                        'fix_values': {
                            'marker_size': 0.04,        # 40mm
                            'marker_separation': 0.01   # 10mm
                        }
                    }
                ],
                'parameter_ranges': {
                    'dictionary_id': {
                        'min': 0,
                        'max': 20,
                        'default_value': 10
                    },
                    'markers_x': {
                        'min': 2,
                        'max': 20,
                        'default_value': 5
                    },
                    'markers_y': {
                        'min': 2, 
                        'max': 20,
                        'default_value': 7
                    }
                },
                'default_corrections': {
                    'dictionary_id': 10,           # DICT_6X6_250
                    'marker_size': 0.04,
                    'marker_separation': 0.01
                }
            }
        }
    }
    
    def __init__(self, markers_x: int, markers_y: int, marker_size: float, 
                 marker_separation: float, dictionary_id: int = cv2.aruco.DICT_6X6_250, 
                 is_planar: bool = True):
        """
        Initialize ArUco Grid Board.
        
        Args:
            markers_x: Number of markers along X-axis
            markers_y: Number of markers along Y-axis  
            marker_size: Physical size of each marker in meters
            marker_separation: Physical separation between markers in meters
            dictionary_id: ArUco dictionary to use
            is_planar: Whether the pattern lies in a plane (True) or has 3D structure (False)
        """
        super().__init__(
            pattern_id="grid_board",
            name="ArUco Grid Board",
            description="Grid of ArUco markers for robust camera calibration",
            is_planar=is_planar
        )
        
        # Use base class validation
        self.validate_dimensions(markers_x, markers_y, min_size=2)
        self.validate_physical_size(marker_size, "marker_size")
        self.validate_physical_size(marker_separation, "marker_separation")
        
        self.markers_x = markers_x
        self.markers_y = markers_y
        self.marker_size = marker_size
        self.marker_separation = marker_separation
        self.dictionary_id = dictionary_id
        
        # Create ArUco dictionary and Grid board
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        
        # Try different Grid board creation methods for different OpenCV versions
        try:
            # OpenCV 4.7+ method
            self.grid_board = cv2.aruco.GridBoard(
                (markers_x, markers_y), marker_size, marker_separation, self.aruco_dict
            )
        except (AttributeError, TypeError):
            try:
                # Older OpenCV method
                self.grid_board = cv2.aruco.GridBoard_create(
                    markers_x, markers_y, marker_size, marker_separation, self.aruco_dict
                )
            except AttributeError:
                raise ValueError("ArUco Grid boards are not supported in this OpenCV version")
        
        # Detector parameters
        try:
            # OpenCV 4.7+ method
            self.detector_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            # Older OpenCV method
            self.detector_params = cv2.aruco.DetectorParameters_create()
        
    @classmethod
    def get_configuration_schema(cls):
        """
        Get the configuration schema for this pattern type.
        
        Returns:
            Dict containing the pattern configuration schema
        """
        return {
            "name": "ArUco Grid Board",
            "description": "Grid of ArUco markers for robust camera calibration detection",
            "icon": "🎲",
            "parameters": [
                {
                    "name": "markers_x",
                    "label": "Markers (X-axis)",
                    "type": "integer", 
                    "default": 5,
                    "min": 2,
                    "max": 20,
                    "description": "Number of markers along X-axis"
                },
                {
                    "name": "markers_y",
                    "label": "Markers (Y-axis)", 
                    "type": "integer",
                    "default": 7,
                    "min": 2,
                    "max": 20,
                    "description": "Number of markers along Y-axis"
                },
                {
                    "name": "marker_size",
                    "label": "Marker Size (meters)",
                    "type": "float",
                    "default": 0.040,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.001,
                    "description": "Physical size of each ArUco marker in meters"
                },
                {
                    "name": "marker_separation",
                    "label": "Marker Separation (meters)",
                    "type": "float",
                    "default": 0.010,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.001,
                    "description": "Physical separation between markers in meters"
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
        
    def detect_corners(self, image: np.ndarray, **kwargs) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect ArUco markers in the grid board."""
        # Use base class utility to convert to grayscale
        gray = self.convert_to_grayscale(image)
            
        # Detect ArUco markers
        try:
            # OpenCV 4.7+ method
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            # Older OpenCV method
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.detector_params
            )
        
        if corners is not None and len(corners) > 0 and ids is not None:
            # For grid board, return corners properly organized by marker ID
            # corners is a list of arrays, each array is [1, 4, 2] for one marker's 4 corners
            # ids is [N, 1] array
            
            # Sort by marker ID to ensure consistent order
            sorted_indices = np.argsort(ids.flatten())
            sorted_corners = [corners[i] for i in sorted_indices]
            sorted_ids = ids.flatten()[sorted_indices]
            
            # Convert to proper format: flatten all corners into [N*4, 2] array
            all_corners = []
            for corner_set in sorted_corners:
                # corner_set shape is [1, 4, 2], we want the [4, 2] part
                marker_corners = corner_set[0]  # Remove the first dimension
                for corner in marker_corners:
                    all_corners.append(corner)
            
            image_points = np.array(all_corners, dtype=np.float32)
            marker_ids = sorted_ids
            
            return True, image_points, marker_ids
        
        return False, None, None
    
    def generate_object_points(self, point_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate 3D object points for ArUco Grid board."""
        # Get Grid board corner positions
        try:
            # Try newer OpenCV method to get all marker corners
            if hasattr(self.grid_board, 'getObjPoints'):
                all_objp = self.grid_board.getObjPoints()
                # Check if it's a tuple or array
                if isinstance(all_objp, tuple):
                    all_objp = all_objp[0]  # Take first element if it's a tuple
                all_objp = np.array(all_objp, dtype=np.float32)
            elif hasattr(self.grid_board, 'objPoints'):
                all_objp = self.grid_board.objPoints
                all_objp = np.array(all_objp, dtype=np.float32)
            else:
                # Manual generation as fallback
                all_objp = self._generate_manual_object_points()
        except AttributeError:
            # Fallback for older OpenCV versions
            all_objp = self._generate_manual_object_points()
        
        # For Grid board, we need to return only the object points corresponding to detected markers
        if point_ids is not None and len(point_ids) > 0:
            # Each ArUco marker has 4 corners, so we need to map marker IDs to corner indices
            corner_indices = []
            for marker_id in point_ids.flatten():
                # Each marker contributes 4 corners (marker_id * 4, marker_id * 4 + 1, ...)
                for i in range(4):
                    corner_indices.append(marker_id * 4 + i)
            
            # Ensure all_objp is a proper numpy array and check bounds
            all_objp = np.array(all_objp, dtype=np.float32)
            if len(all_objp.shape) == 1:
                # Reshape if it's flattened
                total_points = len(all_objp) // 3
                all_objp = all_objp.reshape(total_points, 3)
            
            if len(corner_indices) > 0 and max(corner_indices) < len(all_objp):
                return all_objp[corner_indices]
            else:
                # If indices are out of range, return manual generation
                return self._generate_manual_object_points_for_ids(point_ids)
        else:
            # Return all possible object points
            all_objp = np.array(all_objp, dtype=np.float32)
            if len(all_objp.shape) == 1:
                # Reshape if it's flattened
                total_points = len(all_objp) // 3
                all_objp = all_objp.reshape(total_points, 3)
            return all_objp
    
    def _generate_manual_object_points(self) -> np.ndarray:
        """Generate object points manually for grid board."""
        total_markers = self.markers_x * self.markers_y
        # Each marker has 4 corners
        objp = np.zeros((total_markers * 4, 3), np.float32)
        
        marker_idx = 0
        for j in range(self.markers_y):
            for i in range(self.markers_x):
                # Calculate marker center position
                center_x = i * (self.marker_size + self.marker_separation)
                center_y = j * (self.marker_size + self.marker_separation)
                
                # Calculate 4 corners of the marker (clockwise from top-left)
                half_size = self.marker_size / 2
                corners = [
                    [center_x - half_size, center_y - half_size, 0],  # top-left
                    [center_x + half_size, center_y - half_size, 0],  # top-right
                    [center_x + half_size, center_y + half_size, 0],  # bottom-right
                    [center_x - half_size, center_y + half_size, 0],  # bottom-left
                ]
                
                # Add corners to object points
                for k, corner in enumerate(corners):
                    objp[marker_idx * 4 + k] = corner
                
                marker_idx += 1
        
        return objp
    
    def _generate_manual_object_points_for_ids(self, point_ids: np.ndarray) -> np.ndarray:
        """Generate object points for specific marker IDs."""
        num_detected = len(point_ids)
        objp = np.zeros((num_detected * 4, 3), np.float32)
        
        for idx, marker_id in enumerate(point_ids.flatten()):
            # Calculate marker position from ID
            i = marker_id % self.markers_x
            j = marker_id // self.markers_x
            
            # Calculate marker center position
            center_x = i * (self.marker_size + self.marker_separation)
            center_y = j * (self.marker_size + self.marker_separation)
            
            # Calculate 4 corners of the marker
            half_size = self.marker_size / 2
            corners = [
                [center_x - half_size, center_y - half_size, 0],
                [center_x + half_size, center_y - half_size, 0],
                [center_x + half_size, center_y + half_size, 0],
                [center_x - half_size, center_y + half_size, 0],
            ]
            
            # Add corners to object points
            for k, corner in enumerate(corners):
                objp[idx * 4 + k] = corner
        
        return objp
    
    def get_pattern_size(self) -> Tuple[int, int]:
        """Get pattern size (number of markers)."""
        return (self.markers_x, self.markers_y)
    
    def draw_corners(self, image: np.ndarray, corners: np.ndarray, 
                    point_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw detected ArUco markers with IDs."""
        if corners is not None and len(corners) > 0:
            # Make a copy of the image to avoid modifying the original
            result_image = image.copy()
            
            # corners should be in the format [N*4, 2] from detect_corners
            # We need to reshape it back to [N, 4, 2] for ArUco drawing functions
            if len(corners.shape) == 2 and corners.shape[0] % 4 == 0:
                num_markers = corners.shape[0] // 4
                corners_reshaped = corners.reshape(num_markers, 4, 2)
                
                # Convert to the format expected by cv2.aruco.drawDetectedMarkers
                # It expects a list of arrays, each with shape [1, 4, 2]
                corners_list = [corners_reshaped[i:i+1] for i in range(num_markers)]
                
                # Use OpenCV's built-in ArUco marker drawing function
                try:
                    cv2.aruco.drawDetectedMarkers(result_image, corners_list, point_ids)
                except Exception as e:
                    # Fallback: draw simple circles and lines for corners
                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
                    for i, corner_set in enumerate(corners_reshaped):
                        # Draw corners as circles with different colors
                        for j, corner in enumerate(corner_set):
                            x, y = corner.astype(int)
                            cv2.circle(result_image, (x, y), 5, colors[j % 4], -1)
                            cv2.putText(result_image, str(j), (x+8, y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[j % 4], 1)
                        
                        # Draw marker outline
                        pts = corner_set.astype(int).reshape((-1, 1, 2))
                        cv2.polylines(result_image, [pts], True, (255, 255, 255), 2)
                        
                        # Draw marker ID if available
                        if point_ids is not None and i < len(point_ids):
                            center = corner_set.mean(axis=0).astype(int)
                            cv2.putText(result_image, str(point_ids[i]), 
                                      (center[0]-10, center[1]+5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            else:
                # Fallback: just draw all points as circles
                for i, corner in enumerate(corners):
                    x, y = corner.astype(int)
                    cv2.circle(result_image, (x, y), 5, (0, 255, 0), -1)
            
            return result_image
        return image

    def generate_pattern_image(self, pixels_per_meter: int = None, 
                             border_pixels: int = 50, 
                             pixel_per_square: int = None) -> np.ndarray:
        """
        Generate an ArUco Grid Board pattern image.
        
        Args:
            pixels_per_meter: Number of pixels per meter (default: 2000, i.e., 0.5mm = 1px)
            border_pixels: Border size around the pattern in pixels (default: 50)
            pixel_per_square: Web app compatibility - pixels per square unit (overrides pixels_per_meter)
            
        Returns:
            Generated ArUco Grid Board image as numpy array
        """
        # Handle web app compatibility - convert pixel_per_square to pixels_per_meter
        if pixel_per_square is not None:
            # For grid board, we need to convert from "pixels per square" to "pixels per meter"
            # Assuming the web app treats each "square" as the marker size
            pixels_per_meter = int(pixel_per_square / self.marker_size) if self.marker_size > 0 else 2000
        elif pixels_per_meter is None:
            pixels_per_meter = 2000  # Default value
        # Calculate marker size and separation in pixels
        marker_size_px = int(self.marker_size * pixels_per_meter)
        marker_separation_px = int(self.marker_separation * pixels_per_meter)
        
        # Ensure minimum marker size for OpenCV ArUco generation (at least 20 pixels)
        min_marker_size = 20
        if marker_size_px < min_marker_size:
            print(f"Warning: Marker size {marker_size_px}px too small, using {min_marker_size}px")
            marker_size_px = min_marker_size
        
        # Ensure minimum separation (at least 2 pixels)
        if marker_separation_px < 2:
            marker_separation_px = 2
        
        # Calculate total pattern dimensions
        pattern_width_px = (self.markers_x * marker_size_px + 
                           (self.markers_x - 1) * marker_separation_px)
        pattern_height_px = (self.markers_y * marker_size_px + 
                            (self.markers_y - 1) * marker_separation_px)
        
        # Calculate total image dimensions with borders
        img_width = pattern_width_px + 2 * border_pixels
        img_height = pattern_height_px + 2 * border_pixels
        
        # Use OpenCV's built-in Grid board generation
        try:
            # Method 1: Try new generateImage() method (OpenCV 4.7+)
            if hasattr(self.grid_board, 'generateImage'):
                board_image = self.grid_board.generateImage(
                    (pattern_width_px, pattern_height_px), 
                    marginSize=border_pixels, 
                    borderBits=1
                )
            else:
                # Method 2: Try legacy drawPlanarBoard method
                board_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
                try:
                    cv2.aruco.drawPlanarBoard(
                        self.grid_board, 
                        (img_width, img_height), 
                        board_image, 
                        marginSize=border_pixels, 
                        borderBits=1
                    )
                except AttributeError:
                    # Method 3: Manual fallback
                    return self._generate_manual_grid_pattern(pixels_per_meter, border_pixels)
                
        except (AttributeError, TypeError, cv2.error) as e:
            # Method 3: Manual fallback for very old OpenCV versions or parameter issues
            print(f"Warning: Using manual Grid Board generation due to: {e}")
            return self._generate_manual_grid_pattern(pixels_per_meter, border_pixels)
        
        # Convert to 3-channel BGR if needed
        if len(board_image.shape) == 2:
            board_image = cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)
        
        return board_image
    
    def _generate_manual_grid_pattern(self, pixels_per_meter: int = 1000, 
                                    border_pixels: int = 50) -> np.ndarray:
        """
        Manual Grid Board pattern generation as fallback for older OpenCV versions.
        """
        # Calculate marker size and separation in pixels
        marker_size_px = int(self.marker_size * pixels_per_meter)
        marker_separation_px = int(self.marker_separation * pixels_per_meter)
        
        # Calculate total pattern dimensions
        pattern_width_px = (self.markers_x * marker_size_px + 
                           (self.markers_x - 1) * marker_separation_px)
        pattern_height_px = (self.markers_y * marker_size_px + 
                            (self.markers_y - 1) * marker_separation_px)
        
        # Calculate total image dimensions with borders
        img_width = pattern_width_px + 2 * border_pixels
        img_height = pattern_height_px + 2 * border_pixels
        
        # Create blank white image
        image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # Generate individual markers and place them
        marker_id = 0
        for j in range(self.markers_y):
            for i in range(self.markers_x):
                # Calculate marker position
                x = border_pixels + i * (marker_size_px + marker_separation_px)
                y = border_pixels + j * (marker_size_px + marker_separation_px)
                
                # Generate marker image
                try:
                    marker_img = cv2.aruco.generateImageMarker(
                        self.aruco_dict, marker_id, marker_size_px
                    )
                    # Convert to BGR if needed
                    if len(marker_img.shape) == 2:
                        marker_img = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
                    
                    # Place marker in the image
                    image[y:y+marker_size_px, x:x+marker_size_px] = marker_img
                    
                except Exception as e:
                    # Fallback: draw a simple rectangle with marker ID text
                    cv2.rectangle(image, (x, y), (x+marker_size_px, y+marker_size_px), (0, 0, 0), -1)
                    cv2.rectangle(image, (x+5, y+5), (x+marker_size_px-5, y+marker_size_px-5), (255, 255, 255), -1)
                    cv2.putText(image, str(marker_id), (x+10, y+marker_size_px//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                marker_id += 1
        
        return image
    
    def _get_parameters_dict(self) -> Dict[str, Any]:
        """Get pattern parameters for JSON serialization."""
        return {
            'markers_x': self.markers_x,
            'markers_y': self.markers_y,
            'marker_size': self.marker_size,
            'marker_separation': self.marker_separation,
            'dictionary_id': self.dictionary_id,
            'total_markers': self.markers_x * self.markers_y
        }
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'GridBoard':
        """Create GridBoard from JSON data."""
        params = json_data.get('parameters', {})
        return cls(
            markers_x=params.get('markers_x', 5),
            markers_y=params.get('markers_y', 7),
            marker_size=params.get('marker_size', 0.04),
            marker_separation=params.get('marker_separation', 0.01),
            dictionary_id=params.get('dictionary_id', cv2.aruco.DICT_6X6_250)
        )
