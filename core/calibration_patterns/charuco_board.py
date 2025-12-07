"""
ChArUco Board Pattern
====================

ChArUco board pattern combining chessboard with ArUco markers 
for robust camera calibration detection.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from .base import CalibrationPattern


class CharucoBoard(CalibrationPattern):
    """ChArUco board pattern (chessboard with ArUco markers)."""
    
    # Pattern information for auto-discovery
    PATTERN_INFO = {
        'id': 'charuco_board',
        'name': 'ChArUco Board',
        'icon': '🎯',
        'category': 'chessboard',
        'metadata': {
            'validation_rules': {
                'parameter_relationships': [
                    {
                        'param1': 'square_size',
                        'param2': 'marker_size', 
                        'constraint': 'greater_than',
                        'fix_values': {
                            'square_size': 0.04,  # 40mm
                            'marker_size': 0.02   # 20mm
                        }
                    }
                ],
                'parameter_ranges': {
                    'dictionary_id': {
                        'min': 0,
                        'max': 20,
                        'default_value': 10
                    },
                    'width': {
                        'min': 3,
                        'max': 50,
                        'default_value': 7
                    },
                    'height': {
                        'min': 3, 
                        'max': 50,
                        'default_value': 5
                    }
                },
                'default_corrections': {
                    'dictionary_id': 10,  # DICT_6X6_250
                    'square_size': 0.04,
                    'marker_size': 0.02
                }
            }
        }
    }
    
    def __init__(self, width: int, height: int, square_size: float, marker_size: float, 
                 dictionary_id: int = cv2.aruco.DICT_6X6_250, is_planar: bool = True,
                 first_square_white: bool = False):
        """
        Initialize ChArUco board.
        
        Args:
            width: Number of squares along width
            height: Number of squares along height  
            square_size: Physical size of each square in meters
            marker_size: Physical size of ArUco markers in meters
            dictionary_id: ArUco dictionary to use
            is_planar: Whether the pattern lies in a plane (True) or has 3D structure (False)
            first_square_white: Whether the top-left square should be white (True) or black (False)
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
        self.first_square_white = first_square_white
        
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
                        {"value": cv2.aruco.DICT_6X6_1000, "label": "DICT_6X6_1000"},
                        {"value": cv2.aruco.DICT_7X7_50, "label": "DICT_7X7_50"},
                        {"value": cv2.aruco.DICT_7X7_100, "label": "DICT_7X7_100"},
                        {"value": cv2.aruco.DICT_7X7_250, "label": "DICT_7X7_250"},
                        {"value": cv2.aruco.DICT_7X7_1000, "label": "DICT_7X7_1000"},
                        {"value": cv2.aruco.DICT_ARUCO_ORIGINAL, "label": "DICT_ARUCO_ORIGINAL"},
                        {"value": cv2.aruco.DICT_APRILTAG_16h5, "label": "DICT_APRILTAG_16h5"},
                        {"value": cv2.aruco.DICT_APRILTAG_16H5, "label": "DICT_APRILTAG_16H5"},
                        {"value": cv2.aruco.DICT_APRILTAG_25h9, "label": "DICT_APRILTAG_25h9"},
                        {"value": cv2.aruco.DICT_APRILTAG_25H9, "label": "DICT_APRILTAG_25H9"},
                        {"value": cv2.aruco.DICT_APRILTAG_36h10, "label": "DICT_APRILTAG_36h10"},
                        {"value": cv2.aruco.DICT_APRILTAG_36H10, "label": "DICT_APRILTAG_36H10"},
                        {"value": cv2.aruco.DICT_APRILTAG_36h11, "label": "DICT_APRILTAG_36h11"},
                        {"value": cv2.aruco.DICT_APRILTAG_36H11, "label": "DICT_APRILTAG_36H11"},
                        {"value": cv2.aruco.DICT_ARUCO_MIP_36h12, "label": "DICT_ARUCO_MIP_36h12"},
                        {"value": cv2.aruco.DICT_ARUCO_MIP_36H12, "label": "DICT_ARUCO_MIP_36H12"}
                    ],
                    "description": "ArUco marker dictionary to use"
                },
                {
                    "name": "first_square_white",
                    "label": "First Square White",
                    "type": "boolean",
                    "default": False,
                    "description": "Whether the top-left square should be white (True) or black (False)"
                }
            ]
        }
        
    def detect_corners(self, image: np.ndarray, **kwargs) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect corners using ChArUco detection."""
        # Use base class utility to convert to grayscale
        gray = self.convert_to_grayscale(image)
        
        # For white-first boards, we need custom detection since OpenCV assumes black-first
        if self.first_square_white:
            return self._detect_corners_white_first(gray)
            
        # Standard black-first detection
        # Use newer CharucoDetector if available
        if self.charuco_detector is not None:
            try:
                # OpenCV 4.8+ method using CharucoDetector
                # Configure detector to allow detection with just 1 marker
                charuco_params = self.charuco_detector.getCharucoParameters()
                charuco_params.minMarkers = 1
                self.charuco_detector.setCharucoParameters(charuco_params)
                
                corners_charuco, ids_charuco, corners_aruco, ids_aruco = self.charuco_detector.detectBoard(gray)
                
                # CharucoDetector already includes refinement internally
                if corners_charuco is not None and len(corners_charuco) > 0:
                    return True, corners_charuco, ids_charuco
                else:
                    return False, None, None
            except Exception:
                # Fallback to older method
                pass
        
        # Fallback: Detect ArUco markers first, then interpolate ChArUco corners
        # First, detect ArUco markers
        try:
            # OpenCV 4.7+ method
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
            corners_aruco, ids_aruco, rejected = detector.detectMarkers(gray)
        except AttributeError:
            # Older OpenCV method
            corners_aruco, ids_aruco, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.detector_params
            )
        
        # Refine marker detection before interpolating ChArUco corners
        if len(corners_aruco) > 0 and rejected is not None and len(rejected) > 0:
            try:
                # Try to recover rejected markers for better ChArUco corner detection
                if hasattr(cv2.aruco, 'refineDetectedMarkers'):
                    corners_aruco, ids_aruco, rejected, _ = cv2.aruco.refineDetectedMarkers(
                        gray, self.charuco_board, corners_aruco, ids_aruco, rejected,
                        self.aruco_dict, parameters=self.detector_params
                    )
            except Exception:
                pass  # Refinement not available or failed, continue with original detections
        
        if len(corners_aruco) > 0:
            # Interpolate ChArUco corners from detected ArUco markers
            try:
                # Try different possible function names for interpolation
                interpolate_func = None
                for func_name in ['interpolateCornersCharuco', 'interpolateCorners']:
                    if hasattr(cv2.aruco, func_name):
                        interpolate_func = getattr(cv2.aruco, func_name)
                        break
                
                if interpolate_func:
                    # Set minMarkers to 1 to allow detection with just one marker
                    # Note: interpolateCornersCharuco already includes subpixel refinement internally
                    ret, corners_charuco, ids_charuco = interpolate_func(
                        corners_aruco, ids_aruco, gray, self.charuco_board, minMarkers=1
                    )
                    if ret > 0:
                        return True, corners_charuco, ids_charuco
                        
            except Exception:
                # If interpolation fails, we can't use ChArUco corners
                pass
        
        return False, None, None
    
    def _detect_corners_white_first(self, gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Custom detection for white-first ChArUco boards.
        Uses ArUco marker detection and custom corner interpolation.
        """
        # Detect ArUco markers
        try:
            # OpenCV 4.7+ method
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
            corners_aruco, ids_aruco, _ = detector.detectMarkers(gray)
        except AttributeError:
            # Older OpenCV method
            corners_aruco, ids_aruco, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.detector_params
            )
        
        if ids_aruco is None or len(ids_aruco) == 0:
            return False, None, None
        
        # Build mapping of marker IDs to square positions for white-first layout
        marker_id = 0
        marker_positions = {}  # marker_id -> (row, col)
        
        for row in range(self.height):
            for col in range(self.width):
                # Determine if this square is white (has a marker)
                # Must match the generation logic!
                is_black_in_standard = (row + col) % 2 == 0
                
                # If first_square_white=False: black squares are where (row+col) is even
                # If first_square_white=True: black squares are where (row+col) is odd
                if self.first_square_white:
                    is_black = not is_black_in_standard
                else:
                    is_black = is_black_in_standard
                
                # Markers are placed in white squares
                if not is_black:
                    marker_positions[marker_id] = (row, col)
                    marker_id += 1
        
        # Total number of markers we expect
        max_expected_marker_id = marker_id - 1
        
        # Collect corners with averaging for shared corners
        corner_accumulator = {}  # corner_id -> list of positions
        
        # For each detected marker, calculate the 4 chessboard corners around it
        for i, marker_id in enumerate(ids_aruco.flatten()):
            # Skip markers that are out of range (false positives)
            if marker_id not in marker_positions or marker_id > max_expected_marker_id:
                continue
            
            row, col = marker_positions[marker_id]
            marker_corners = corners_aruco[i][0]  # 4 corners: [TL, TR, BR, BL]
            
            # The marker is centered in the square with known relative size
            # marker_size and square_size define the geometry
            # We need to compute a homography from marker space to square space
            
            # Define the marker corners in normalized marker coordinate system (0,0) to (1,1)
            marker_points_norm = np.array([
                [0, 0],  # Top-left
                [1, 0],  # Top-right
                [1, 1],  # Bottom-right
                [0, 1],  # Bottom-left
            ], dtype=np.float32)
            
            # Define the square corners in normalized square coordinate system (0,0) to (1,1)
            # The marker is centered in the square, so we need to calculate its position
            # marker_size / square_size gives the ratio
            ratio = self.marker_size / self.square_size
            margin = (1.0 - ratio) / 2.0  # Space on each side of marker
            
            # Marker occupies [margin, margin] to [margin+ratio, margin+ratio] in square space
            # We want to map from this to full square [0, 0] to [1, 1]
            square_points_norm = np.array([
                [0, 0],  # Top-left corner of square
                [1, 0],  # Top-right corner of square
                [1, 1],  # Bottom-right corner of square
                [0, 1],  # Bottom-left corner of square
            ], dtype=np.float32)
            
            # The marker corners in square coordinate system
            marker_in_square = np.array([
                [margin, margin],                    # TL marker in square space
                [margin + ratio, margin],            # TR marker in square space
                [margin + ratio, margin + ratio],    # BR marker in square space
                [margin, margin + ratio],            # BL marker in square space
            ], dtype=np.float32)
            
            # Compute homography from marker corners (in image) to square corners (in image)
            # We have: marker_corners (image space) corresponds to marker_in_square (normalized square space)
            # We want: square corners in image space
            # So we compute H: marker_in_square -> marker_corners (image)
            # Then use H to transform square_points_norm to get square corners in image
            
            try:
                # Compute homography from normalized marker-in-square coords to detected marker image coords
                H, _ = cv2.findHomography(marker_in_square, marker_corners)
                
                if H is not None:
                    # Transform the square corner positions to image space
                    square_corners_homog = cv2.perspectiveTransform(
                        square_points_norm.reshape(1, -1, 2), H
                    )
                    square_corners = square_corners_homog[0]  # Shape: (4, 2)
                else:
                    # Fallback: use simple scaling from center
                    marker_center = np.mean(marker_corners, axis=0)
                    square_corners = []
                    for mc in marker_corners:
                        vec = mc - marker_center
                        sc = marker_center + vec / ratio
                        square_corners.append(sc)
                    square_corners = np.array(square_corners)
            except:
                # Fallback: use simple scaling from center
                marker_center = np.mean(marker_corners, axis=0)
                square_corners = []
                for mc in marker_corners:
                    vec = mc - marker_center
                    sc = marker_center + vec / ratio
                    square_corners.append(sc)
                square_corners = np.array(square_corners)
            
            # Map the 4 corners of each detected marker to their corresponding chessboard corner IDs
            # Each marker is located in a square at position (row, col)
            # ChArUco boards have interior corners only (like standard chessboards)
            # Total corners = (width-1) * (height-1)
            # 
            # Corner ID mapping:
            # - A square at (row, col) has 4 corners at grid intersections
            # - These intersections are at positions (row, col), (row, col+1), (row+1, col+1), (row+1, col)
            # - Only interior intersections are valid ChArUco corners
            # - Interior means: 0 < intersection_row < height and 0 < intersection_col < width
            # - Corner ID for intersection (r, c) = (r-1) * (width-1) + (c-1)
            
            corner_infos = [
                # (square_corner_index, intersection_row, intersection_col)
                (0, row, col),           # Top-left corner of square
                (1, row, col + 1),       # Top-right corner of square
                (2, row + 1, col + 1),   # Bottom-right corner of square
                (3, row + 1, col),       # Bottom-left corner of square
            ]
            
            for sc_idx, int_row, int_col in corner_infos:
                # Check if this intersection is an interior corner (not on board boundary)
                if 0 < int_row < self.height and 0 < int_col < self.width:
                    # Calculate the ChArUco corner ID (0-indexed from interior corners)
                    corner_id = (int_row - 1) * (self.width - 1) + (int_col - 1)
                    
                    if corner_id not in corner_accumulator:
                        corner_accumulator[corner_id] = []
                    corner_accumulator[corner_id].append(square_corners[sc_idx])
        
        if len(corner_accumulator) == 0:
            return False, None, None
        
        # Average corners that were detected multiple times (shared by multiple markers)
        final_corners = {}
        for corner_id, positions in corner_accumulator.items():
            final_corners[corner_id] = np.mean(positions, axis=0)
        
        # Convert to numpy arrays
        final_ids = np.array(list(final_corners.keys()), dtype=np.int32).reshape(-1, 1)
        final_corners_array = np.array([final_corners[cid] for cid in final_ids.flatten()], dtype=np.float32).reshape(-1, 1, 2)
        
        # Apply subpixel corner refinement for better accuracy
        if len(final_corners_array) > 0:
            # Build a map of all marker corner positions (matching OpenCV approach)
            marker_corners_flat = []  # All marker corner positions
            if len(corners_aruco) > 0:
                for marker_corners in corners_aruco:
                    corners_flat = marker_corners[0]
                    for corner in corners_flat:
                        marker_corners_flat.append(corner)
            
            # Define criteria for subpixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            
            # Store refinement stages for visualization
            refinement_stages = []  # List of (corner_id, [initial, refined], [window], [movement])
            
            # Refine each corner
            corners_to_keep = []
            ids_to_keep = []
            
            for i, corner in enumerate(final_corners_array):
                corner_id = final_ids[i][0]
                cx, cy = corner[0]
                
                # Store initial position
                initial_pos = (float(cx), float(cy))
                
                # Find distance to nearest marker corner
                min_dist_to_marker = float('inf')
                if marker_corners_flat:
                    for marker_corner in marker_corners_flat:
                        dist = np.sqrt((cx - marker_corner[0])**2 + (cy - marker_corner[1])**2)
                        if dist < min_dist_to_marker:
                            min_dist_to_marker = dist
                
                # Window size is half the distance to nearest marker corner (matching OpenCV)
                if min_dist_to_marker != float('inf'):
                    window_size_px = int(min_dist_to_marker / 2.0)
                    # Ensure window size is odd and at least 3
                    window_size_px = max(3, window_size_px if window_size_px % 2 == 1 else window_size_px + 1)
                else:
                    window_size_px = 5
                
                # Check boundaries - skip corner if window doesn't fit within image
                half_win = window_size_px // 2
                min_x = half_win
                max_x = gray.shape[1] - half_win - 1
                min_y = half_win
                max_y = gray.shape[0] - half_win - 1
                
                # Skip this corner if it's too close to the border
                if cx < min_x or cx > max_x or cy < min_y or cy > max_y:
                    continue
                
                # Store position before refinement
                prev_cx, prev_cy = cx, cy
                
                # Adjust coordinates by -0.5 pixels before cornerSubPix (matching OpenCV)
                corner_adjusted = np.float32([[cx - 0.5, cy - 0.5]])
                cv2.cornerSubPix(gray, corner_adjusted, (window_size_px, window_size_px), (-1, -1), criteria)
                
                # Add back 0.5 pixels after refinement (matching OpenCV)
                cx, cy = corner_adjusted[0] + np.float32([0.5, 0.5])
                refined_pos = (float(cx), float(cy))
                
                # Calculate how much the corner moved
                movement = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                
                # Update corner position
                corner[0] = [cx, cy]
                
                # Keep this corner
                corners_to_keep.append(corner)
                ids_to_keep.append(corner_id)
                
                # Store refinement stage
                refinement_stages.append((corner_id, [initial_pos, refined_pos], [window_size_px], [float(movement)]))
            
            # Update arrays with only valid corners
            if len(corners_to_keep) > 0:
                final_corners_array = np.array(corners_to_keep, dtype=np.float32)
                final_ids = np.array(ids_to_keep, dtype=np.int32).reshape(-1, 1)
            else:
                final_corners_array = None
                final_ids = None
            
            # Store refinement stages as an attribute for visualization
            self._last_refinement_stages = refinement_stages
        
        if final_corners_array is None or len(final_corners_array) == 0:
            return False, None, None
        
        return True, final_corners_array, final_ids
    
    def generate_object_points(self, point_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate 3D object points for ChArUco board."""
        # Generate all corner 3D positions
        # ChArUco corners are at the intersections of squares
        all_objp = np.zeros(((self.width - 1) * (self.height - 1), 3), np.float32)
        for row in range(self.height - 1):
            for col in range(self.width - 1):
                corner_idx = row * (self.width - 1) + col
                # Corner position in 3D space
                all_objp[corner_idx] = [col * self.square_size, row * self.square_size, 0.0]
        
        # Return object points for detected corners
        if point_ids is not None and len(point_ids) > 0:
            return all_objp[point_ids.flatten()]
        else:
            return all_objp
    
    def get_pattern_size(self) -> Tuple[int, int]:
        """Get pattern size (internal corners)."""
        return (self.width - 1, self.height - 1)
    
    def draw_corners(self, image: np.ndarray, corners: np.ndarray, 
                    point_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw detected ChArUco corners with IDs using OpenCV."""
        if corners is not None and len(corners) > 0:
            # Make a copy of the image to avoid modifying the original
            result_image = image.copy()
            
            # Use OpenCV's built-in ChArUco corner drawing function
            if point_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(result_image, corners, point_ids, (0, 0, 255))
            else:
                # If no point_ids available, just draw red circles
                for corner in corners:
                    x, y = corner.ravel().astype(int)
                    cv2.circle(result_image, (x, y), 5, (0, 0, 255), -1)
            
            # Draw single pixel markers for precise localization in blue
            for corner in corners:
                x, y = corner.ravel()
                x, y = int(round(x)), int(round(y))
                
                # Draw single pixel in blue
                if 0 <= x < result_image.shape[1] and 0 <= y < result_image.shape[0]:
                    result_image[y, x] = (255, 0, 0)
            
            return result_image
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
        # For white-first boards, use manual generation
        if self.first_square_white:
            return self._generate_manual_charuco_pattern(pixel_per_square, border_pixels)
        
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
        Generate ChArUco pattern with markers properly placed in white squares.
        Handles both first_square_white=True and first_square_white=False.
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
        marker_size_px = int(pixel_per_square * marker_ratio)
        
        marker_id = 0
        
        # Draw ChArUco pattern
        for row in range(self.height):
            for col in range(self.width):
                # Calculate square position
                x1 = int(start_x + col * pixel_per_square)
                y1 = int(start_y + row * pixel_per_square)
                x2 = int(start_x + (col + 1) * pixel_per_square)
                y2 = int(start_y + (row + 1) * pixel_per_square)
                
                # Determine if this square should be black or white
                # Standard chessboard: (row + col) % 2 == 0 means black
                is_black_in_standard = (row + col) % 2 == 0
                
                # If first_square_white=False (default): top-left is black, so black squares are where (row+col) is even
                # If first_square_white=True: top-left is white, so black squares are where (row+col) is odd
                if self.first_square_white:
                    is_black = not is_black_in_standard
                else:
                    is_black = is_black_in_standard
                
                if is_black:
                    # Draw black square
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
                else:
                    # White square - place ArUco marker here
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
                    
                    # Generate and draw the actual ArUco marker
                    if marker_id < len(self.aruco_dict.bytesList):
                        # Generate marker image
                        marker_img = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, marker_size_px)
                        
                        # Calculate position to center marker in square
                        marker_x1 = int(x1 + (pixel_per_square - marker_size_px) / 2)
                        marker_y1 = int(y1 + (pixel_per_square - marker_size_px) / 2)
                        marker_x2 = marker_x1 + marker_size_px
                        marker_y2 = marker_y1 + marker_size_px
                        
                        # Convert marker to BGR if needed
                        if len(marker_img.shape) == 2:
                            marker_img = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
                        
                        # Place marker in the white square
                        image[marker_y1:marker_y2, marker_x1:marker_x2] = marker_img
                    
                    marker_id += 1
        
        return image
    
    def _get_parameters_dict(self) -> Dict[str, Any]:
        """Get pattern parameters for JSON serialization."""
        return {
            'width': self.width,
            'height': self.height,
            'square_size': self.square_size,
            'marker_size': self.marker_size,
            'dictionary_id': self.dictionary_id,
            'first_square_white': self.first_square_white,
            'total_corners': (self.width - 1) * (self.height - 1)
        }
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'CharucoBoard':
        """Create CharucoBoard from JSON data."""
        params = json_data.get('parameters', {})
        return cls(
            width=params.get('width', 9),
            height=params.get('height', 7),
            square_size=params.get('square_size', 0.025),
            marker_size=params.get('marker_size', 0.015),
            dictionary_id=params.get('dictionary_id', cv2.aruco.DICT_6X6_250),
            is_planar=json_data.get('is_planar', True),
            first_square_white=params.get('first_square_white', False)
        )
