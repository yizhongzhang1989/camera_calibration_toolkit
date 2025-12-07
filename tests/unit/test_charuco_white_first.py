#!/usr/bin/env python3
"""
Unit Tests for ChArUco Board with White First Square
====================================================

This module contains comprehensive unit tests for the white-first square functionality
in ChArUco board pattern generation and detection.

Test Classes:
1. TestCharucoWhiteFirst:
   - Tests pattern generation with first_square_white=True
   - Validates corner detection on generated patterns
   - Compares detected corners with known/expected positions
   - Tests accuracy of subpixel refinement with border filtering

The white-first ChArUco functionality is used in:
- Custom calibration board generation
- Camera intrinsic calibration with specialized board layouts
- Pattern detection with improved corner localization
"""

import unittest
import os
import sys
import numpy as np
import cv2
from typing import Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.calibration_patterns import get_pattern_manager, CharucoBoard


class TestCharucoWhiteFirst(unittest.TestCase):
    """Test ChArUco board with white first square functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = get_pattern_manager()
        
        # Standard test configuration
        self.test_width = 9
        self.test_height = 6
        self.test_square_size = 0.04
        self.test_marker_size = 0.03
        self.test_dict = cv2.aruco.DICT_4X4_100
        
        # Image generation parameters
        self.image_size = (1200, 900)  # Width x Height
        self.pixel_per_square = 100  # Pixels per square

    def _generate_board_image(
        self,
        board: CharucoBoard,
        image_size: Tuple[int, int] = None,
        pixel_per_square: int = None
    ) -> Tuple[np.ndarray, int, int]:
        """
        Generate a synthetic image of the ChArUco board.
        
        Args:
            board: CharucoBoard instance
            image_size: (width, height) of output image in pixels
            pixel_per_square: Size of each square in pixels
            
        Returns:
            Tuple of (generated image, x_offset, y_offset)
        """
        if image_size is None:
            image_size = self.image_size
        if pixel_per_square is None:
            pixel_per_square = self.pixel_per_square
            
        # Generate the board pattern image
        board_img = board.generate_pattern_image(
            pixel_per_square=pixel_per_square,
            border_pixels=100
        )
        
        # Convert to grayscale if needed
        if len(board_img.shape) == 3:
            board_img = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
        
        # Resize or pad to desired size if needed
        if board_img.shape[1] > image_size[0] or board_img.shape[0] > image_size[1]:
            # Scale down if too large
            scale = min(image_size[0] / board_img.shape[1], image_size[1] / board_img.shape[0])
            new_width = int(board_img.shape[1] * scale)
            new_height = int(board_img.shape[0] * scale)
            board_img = cv2.resize(board_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create canvas and center the board
        canvas = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 255
        y_offset = (canvas.shape[0] - board_img.shape[0]) // 2
        x_offset = (canvas.shape[1] - board_img.shape[1]) // 2
        canvas[y_offset:y_offset+board_img.shape[0], x_offset:x_offset+board_img.shape[1]] = board_img
        
        return canvas, x_offset, y_offset

    def _compute_expected_corners(
        self,
        board: CharucoBoard,
        x_offset: int,
        y_offset: int,
        pixel_per_square: int = None
    ) -> dict:
        """
        Compute expected corner positions in image coordinates.
        
        Args:
            board: CharucoBoard instance
            x_offset: X offset of board in image
            y_offset: Y offset of board in image
            pixel_per_square: Size of each square in pixels
            
        Returns:
            Dictionary mapping corner_id to (x, y) position
        """
        if pixel_per_square is None:
            pixel_per_square = self.pixel_per_square
            
        expected_corners = {}
        border_pixels = 100
        
        # ChArUco corners are at square intersections
        for row in range(board.height - 1):
            for col in range(board.width - 1):
                corner_id = row * (board.width - 1) + col
                
                # Position in pixels from top-left of board
                x_px = (col + 1) * pixel_per_square + border_pixels + x_offset
                y_px = (row + 1) * pixel_per_square + border_pixels + y_offset
                
                expected_corners[corner_id] = (x_px, y_px)
        
        return expected_corners

    def test_white_first_pattern_generation(self):
        """Test that ChArUco board can generate pattern with first square white."""
        # Create board with first_square_white=True
        board = self.manager.create_pattern(
            'charuco_board',
            width=self.test_width,
            height=self.test_height,
            square_size=self.test_square_size,
            marker_size=self.test_marker_size,
            dictionary_id=self.test_dict,
            first_square_white=True
        )
        
        # Verify parameter is set
        self.assertTrue(board.first_square_white)
        
        # Generate image
        image, x_offset, y_offset = self._generate_board_image(board)
        
        # Verify image was generated
        self.assertIsNotNone(image)
        self.assertEqual(image.shape, (self.image_size[1], self.image_size[0]))
        
        # Check that first square (top-left) is white
        # Sample a point inside the first square
        sample_x = x_offset + 110  # Inside first square (border=100, offset a bit)
        sample_y = y_offset + 110
        self.assertGreater(image[sample_y, sample_x], 200, "First square should be white")

    def test_white_first_corner_detection(self):
        """Test that corners are correctly detected on white-first ChArUco board."""
        # Create board with first_square_white=True
        board = self.manager.create_pattern(
            'charuco_board',
            width=self.test_width,
            height=self.test_height,
            square_size=self.test_square_size,
            marker_size=self.test_marker_size,
            dictionary_id=self.test_dict,
            first_square_white=True
        )
        
        # Generate image
        image, x_offset, y_offset = self._generate_board_image(board)
        
        # Detect corners
        success, corners, ids = board.detect_corners(image)
        
        # Verify detection succeeded
        self.assertTrue(success, "Corner detection should succeed")
        self.assertIsNotNone(corners, "Corners should not be None")
        self.assertIsNotNone(ids, "IDs should not be None")
        self.assertGreater(len(corners), 0, "Should detect at least some corners")
        
        # Verify corners and ids have matching lengths
        self.assertEqual(len(corners), len(ids))

    def test_detected_corners_match_expected_positions(self):
        """Test that detected corner positions match expected/known positions."""
        # Test configurations: (width, height) pairs
        board_configs = [
            (3, 3), (4, 3), (5, 3), (6, 3),
            (3, 4), (4, 4), (5, 4), (6, 4),
            (3, 5), (4, 5), (5, 5), (6, 5),
            (3, 6), (4, 6), (5, 6), (6, 6)
        ]
        
        # Marker size ratios to test
        marker_ratios = [0.3, 0.5, 0.7, 0.8]
        
        print("\nCorner Detection Accuracy Across Different Board Sizes and Marker Ratios:")
        print("=" * 90)
        
        for marker_ratio in marker_ratios:
            print(f"\nMarker Size Ratio: {marker_ratio} (marker_size = {marker_ratio} * square_size)")
            print("-" * 90)
            
            for width, height in board_configs:
                marker_size = self.test_square_size * marker_ratio
                
                with self.subTest(width=width, height=height, marker_ratio=marker_ratio):
                    # Create board with first_square_white=True
                    board = self.manager.create_pattern(
                        'charuco_board',
                        width=width,
                        height=height,
                        square_size=self.test_square_size,
                        marker_size=marker_size,
                        dictionary_id=self.test_dict,
                        first_square_white=True
                    )
                    
                    # Generate image
                    image, x_offset, y_offset = self._generate_board_image(board)
                    
                    # Compute expected corner positions
                    expected_corners = self._compute_expected_corners(board, x_offset, y_offset)
                    
                    # Detect corners
                    success, corners, ids = board.detect_corners(image)
                    
                    self.assertTrue(success, 
                                  f"Detection failed for {width}x{height} board with ratio {marker_ratio}")
                    self.assertIsNotNone(corners)
                    self.assertIsNotNone(ids)
                    
                    # Compare detected corners with expected positions
                    max_error = 0.0
                    errors = []
                    
                    for i, corner_id in enumerate(ids.flatten()):
                        if corner_id in expected_corners:
                            detected_pos = corners[i][0]
                            expected_pos = expected_corners[corner_id]
                            
                            # Calculate Euclidean distance
                            error = np.sqrt(
                                (detected_pos[0] - expected_pos[0])**2 +
                                (detected_pos[1] - expected_pos[1])**2
                            )
                            errors.append(error)
                            max_error = max(max_error, error)
                    
                    # Verify all detected corners have expected positions
                    self.assertEqual(len(errors), len(corners), 
                                   f"All detected corners should have expected positions for {width}x{height} ratio {marker_ratio}")
                    
                    # Calculate statistics
                    mean_error = np.mean(errors) if errors else 0.0
                    std_error = np.std(errors) if errors else 0.0
                    
                    # Assert accuracy - corners should be within 2 pixels of expected position
                    # (synthetic images should have very accurate detection)
                    self.assertLess(mean_error, 2.0, 
                                  f"Mean error {mean_error:.2f} pixels is too high for {width}x{height} ratio {marker_ratio}")
                    self.assertLess(max_error, 5.0, 
                                  f"Max error {max_error:.2f} pixels is too high for {width}x{height} ratio {marker_ratio}")
                    
                    # Print statistics for this configuration
                    total_corners = (width - 1) * (height - 1)
                    print(f"  {width}x{height}: {len(corners):2d}/{total_corners:2d} corners | "
                          f"Mean: {mean_error:.3f}px | Std: {std_error:.3f}px | Max: {max_error:.3f}px")
        
        print("=" * 90)

    def test_border_corner_filtering(self):
        """Test that corners near image borders are filtered out when window doesn't fit."""
        # Create board with first_square_white=True
        board = self.manager.create_pattern(
            'charuco_board',
            width=self.test_width,
            height=self.test_height,
            square_size=self.test_square_size,
            marker_size=self.test_marker_size,
            dictionary_id=self.test_dict,
            first_square_white=True
        )
        
        # Generate image with minimal border to force some corners near edges
        board_img = board.generate_pattern_image(
            pixel_per_square=self.pixel_per_square,
            border_pixels=20  # Small border
        )
        
        # Detect corners
        success, corners, ids = board.detect_corners(board_img)
        
        if success and corners is not None:
            # Verify that no corner is too close to the border
            # (corners near border should be filtered out)
            min_distance_from_border = float('inf')
            
            for corner in corners:
                x, y = corner[0]
                dist_to_border = min(
                    x,  # distance to left
                    board_img.shape[1] - x,  # distance to right
                    y,  # distance to top
                    board_img.shape[0] - y   # distance to bottom
                )
                min_distance_from_border = min(min_distance_from_border, dist_to_border)
            
            # With border filtering, minimum distance should be reasonable
            # (at least half the window size, which is typically ~30-50 pixels)
            self.assertGreater(
                min_distance_from_border, 15,
                "Corners too close to border should be filtered out"
            )

    def test_comparison_white_first_vs_black_first(self):
        """Test that white-first and black-first boards detect same number of corners."""
        # Create two boards with different first square colors
        board_white_first = self.manager.create_pattern(
            'charuco_board',
            width=self.test_width,
            height=self.test_height,
            square_size=self.test_square_size,
            marker_size=self.test_marker_size,
            dictionary_id=self.test_dict,
            first_square_white=True
        )
        
        board_black_first = self.manager.create_pattern(
            'charuco_board',
            width=self.test_width,
            height=self.test_height,
            square_size=self.test_square_size,
            marker_size=self.test_marker_size,
            dictionary_id=self.test_dict,
            first_square_white=False
        )
        
        # Generate images
        image_white, _, _ = self._generate_board_image(board_white_first)
        image_black, _, _ = self._generate_board_image(board_black_first)
        
        # Detect corners on both
        success_white, corners_white, ids_white = board_white_first.detect_corners(image_white)
        success_black, corners_black, ids_black = board_black_first.detect_corners(image_black)
        
        # Both should succeed
        self.assertTrue(success_white)
        self.assertTrue(success_black)
        
        # Both should detect similar number of corners
        # (may vary slightly due to marker placement, but should be close)
        self.assertGreater(len(corners_white), 0)
        self.assertGreater(len(corners_black), 0)
        
        # Check that they detect a reasonable number of corners
        total_corners = (self.test_width - 1) * (self.test_height - 1)
        self.assertGreater(
            len(corners_white), total_corners * 0.5,
            "Should detect at least 50% of corners"
        )
        self.assertGreater(
            len(corners_black), total_corners * 0.5,
            "Should detect at least 50% of corners"
        )

    def test_serialization_with_white_first(self):
        """Test that first_square_white parameter is preserved in serialization."""
        # Create board with first_square_white=True
        original_board = self.manager.create_pattern(
            'charuco_board',
            width=self.test_width,
            height=self.test_height,
            square_size=self.test_square_size,
            marker_size=self.test_marker_size,
            dictionary_id=self.test_dict,
            first_square_white=True
        )
        
        # Serialize to JSON
        pattern_dict = original_board.to_json()
        
        # Verify first_square_white is in parameters
        self.assertIn('first_square_white', pattern_dict['parameters'])
        self.assertTrue(pattern_dict['parameters']['first_square_white'])
        
        # Deserialize
        from core.calibration_patterns import load_pattern_from_json
        restored_board = load_pattern_from_json(pattern_dict)
        
        # Verify parameter is preserved
        self.assertTrue(restored_board.first_square_white)

    def test_multiple_board_sizes(self):
        """Test white-first detection works with various board sizes."""
        test_configs = [
            (5, 4),   # Small
            (9, 6),   # Medium
            (12, 9),  # Large
        ]
        
        for width, height in test_configs:
            with self.subTest(width=width, height=height):
                board = self.manager.create_pattern(
                    'charuco_board',
                    width=width,
                    height=height,
                    square_size=self.test_square_size,
                    marker_size=self.test_marker_size,
                    dictionary_id=self.test_dict,
                    first_square_white=True
                )
                
                # Generate image
                image, _, _ = self._generate_board_image(board)
                
                # Detect corners
                success, corners, ids = board.detect_corners(image)
                
                # Verify detection
                self.assertTrue(success, f"Detection failed for {width}x{height} board")
                self.assertIsNotNone(corners)
                self.assertGreater(len(corners), 0)

    def test_partial_board_left_visible_right_cropped(self):
        """Test corner detection with left border visible but right side cropped."""
        # Create a 9x4 board as specified in the original issue
        board = self.manager.create_pattern(
            'charuco_board',
            width=9,
            height=4,
            square_size=0.1148,
            marker_size=0.09,
            dictionary_id=cv2.aruco.DICT_4X4_100,
            first_square_white=True
        )
        
        # Generate full board image
        full_image = board.generate_pattern_image(pixel_per_square=100, border_pixels=50)
        h, w = full_image.shape[:2]
        
        # Crop to keep left 60% (left border visible, right cropped)
        cropped_image = full_image[:, :int(w*0.6)]
        
        # Detect corners
        success, corners, ids = board.detect_corners(cropped_image)
        
        # Should successfully detect corners
        self.assertTrue(success, "Should detect corners on partially visible board")
        self.assertIsNotNone(corners)
        self.assertGreater(len(corners), 0, "Should detect at least some corners")
        
        # Critical test: corners should NOT be shifted to image boundaries
        corner_positions = corners.reshape(-1, 2)
        crop_h, crop_w = cropped_image.shape[:2]
        
        # Check corners near left boundary (should be properly positioned, not at x=0)
        boundary_threshold = 30
        near_left = np.sum(corner_positions[:, 0] < boundary_threshold)
        
        # With proper homography-based detection, corners should not cluster at boundaries
        # Allow at most 1-2 corners near boundary (these would be actual board edge corners)
        self.assertLessEqual(near_left, 2, 
                           f"Too many corners ({near_left}) near left boundary - corners may be shifting to border")
        
        # Verify corner IDs are consistent (should be sequential interior corners)
        detected_ids = sorted(ids.flatten().tolist())
        self.assertGreater(len(detected_ids), 5, "Should detect multiple corners on partial board")

    def test_partial_board_various_crops(self):
        """Test corner detection with various crop scenarios."""
        board = self.manager.create_pattern(
            'charuco_board',
            width=9,
            height=6,
            square_size=self.test_square_size,
            marker_size=self.test_marker_size,
            dictionary_id=self.test_dict,
            first_square_white=True
        )
        
        # Generate full board
        full_image = board.generate_pattern_image(pixel_per_square=80, border_pixels=50)
        h, w = full_image.shape[:2]
        
        crop_scenarios = [
            ("left_and_top", full_image[int(h*0.3):, int(w*0.3):], "Left & top cropped"),
            ("right_and_bottom", full_image[:int(h*0.7), :int(w*0.7)], "Right & bottom cropped"),
            ("center_crop", full_image[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)], "Center crop"),
        ]
        
        for scenario_name, cropped, description in crop_scenarios:
            with self.subTest(scenario=scenario_name):
                # Detect corners
                success, corners, ids = board.detect_corners(cropped)
                
                # Should detect corners successfully
                self.assertTrue(success, f"Detection should succeed for {description}")
                self.assertIsNotNone(corners)
                self.assertGreater(len(corners), 0, f"Should detect corners for {description}")
                
                # Verify no corners are shifted to boundaries
                corner_positions = corners.reshape(-1, 2)
                crop_h, crop_w = cropped.shape[:2]
                boundary_threshold = 10
                
                near_boundary = np.sum(
                    (corner_positions[:, 0] < boundary_threshold) |
                    (corner_positions[:, 0] > crop_w - boundary_threshold) |
                    (corner_positions[:, 1] < boundary_threshold) |
                    (corner_positions[:, 1] > crop_h - boundary_threshold)
                )
                
                # Should have very few or no corners at boundaries
                corner_ratio = near_boundary / len(corners)
                self.assertLess(corner_ratio, 0.2, 
                              f"Too many corners ({near_boundary}/{len(corners)}) near boundaries for {description}")

    def test_partial_board_corner_accuracy(self):
        """Test that corners on partially visible boards are accurately positioned."""
        board = self.manager.create_pattern(
            'charuco_board',
            width=7,
            height=5,
            square_size=self.test_square_size,
            marker_size=self.test_marker_size,
            dictionary_id=self.test_dict,
            first_square_white=True
        )
        
        # Generate full board with known parameters
        pixel_per_square = 100
        border_pixels = 50
        full_image = board.generate_pattern_image(
            pixel_per_square=pixel_per_square,
            border_pixels=border_pixels
        )
        h, w = full_image.shape[:2]
        
        # Crop to partial view
        partial_image = full_image[int(h*0.2):int(h*0.8), int(w*0.15):int(w*0.85)]
        
        # Detect corners on both full and partial
        success_full, corners_full, ids_full = board.detect_corners(full_image)
        success_partial, corners_partial, ids_partial = board.detect_corners(partial_image)
        
        self.assertTrue(success_full, "Full board detection should succeed")
        self.assertTrue(success_partial, "Partial board detection should succeed")
        
        # Find corners that appear in both detections
        common_ids = set(ids_full.flatten()) & set(ids_partial.flatten())
        self.assertGreater(len(common_ids), 3, "Should have overlapping corners in both detections")
        
        # For common corners, positions should be relatively consistent
        # (accounting for the crop offset)
        crop_offset_x = int(w * 0.15)
        crop_offset_y = int(h * 0.2)
        
        for corner_id in list(common_ids)[:5]:  # Check first 5 common corners
            # Find positions in full image
            idx_full = np.where(ids_full.flatten() == corner_id)[0][0]
            pos_full = corners_full[idx_full][0]
            
            # Find positions in partial image
            idx_partial = np.where(ids_partial.flatten() == corner_id)[0][0]
            pos_partial = corners_partial[idx_partial][0]
            
            # Adjust partial position for crop offset
            pos_partial_adjusted = pos_partial + np.array([crop_offset_x, crop_offset_y])
            
            # Calculate error
            error = np.linalg.norm(pos_full - pos_partial_adjusted)
            
            # Error should be small (within a few pixels)
            self.assertLess(error, 3.0, 
                          f"Corner {corner_id} position inconsistent between full and partial: {error:.2f} pixels")

    def test_partial_board_minimal_markers(self):
        """Test corner detection when only a few markers are visible."""
        board = self.manager.create_pattern(
            'charuco_board',
            width=8,
            height=6,
            square_size=self.test_square_size,
            marker_size=self.test_marker_size,
            dictionary_id=self.test_dict,
            first_square_white=True
        )
        
        # Generate full board
        full_image = board.generate_pattern_image(pixel_per_square=100, border_pixels=50)
        h, w = full_image.shape[:2]
        
        # Aggressive crop to show only corner region (minimal markers)
        corner_crop = full_image[:int(h*0.4), :int(w*0.4)]
        
        # Detect corners
        success, corners, ids = board.detect_corners(corner_crop)
        
        # Even with few markers, detection should work
        if success:
            self.assertIsNotNone(corners)
            self.assertGreater(len(corners), 0, "Should detect some corners even with minimal markers")
            
            # Verify corners are properly positioned (not at boundaries)
            corner_positions = corners.reshape(-1, 2)
            crop_h, crop_w = corner_crop.shape[:2]
            
            # Calculate average distance from boundaries
            distances = []
            for pos in corner_positions:
                dist = min(pos[0], crop_w - pos[0], pos[1], crop_h - pos[1])
                distances.append(dist)
            
            avg_boundary_dist = np.mean(distances)
            # Average distance should be reasonable (corners not clustered at edges)
            self.assertGreater(avg_boundary_dist, 5.0, 
                             "Corners should not be clustered at image boundaries")


if __name__ == '__main__':
    unittest.main()
