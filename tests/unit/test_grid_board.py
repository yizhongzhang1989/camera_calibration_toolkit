#!/usr/bin/env python3
"""
Unit Tests for Marker Grid Pattern
===================================

Tests for GridBoard calibration pattern creation, configuration,
pattern image generation, corner detection, object point generation,
and JSON serialization.
"""

import unittest
import os
import sys
import json
import numpy as np
import cv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.calibration_patterns.grid_board import GridBoard
from core.calibration_patterns import get_pattern_manager, create_pattern_from_json


def _get_supported_aruco_dictionary_ids():
    """Return all predefined ArUco dictionary IDs supported by this OpenCV build."""
    dict_ids = {}
    for name in dir(cv2.aruco):
        if not name.startswith("DICT_"):
            continue

        value = getattr(cv2.aruco, name)
        if not isinstance(value, int):
            continue

        try:
            cv2.aruco.getPredefinedDictionary(value)
            dict_ids[value] = name
        except Exception:
            # Skip aliases or constants not usable as predefined dictionary IDs.
            continue

    return sorted(dict_ids.items(), key=lambda item: item[0])


class TestGridBoardCreation(unittest.TestCase):
    """Test GridBoard creation and parameter validation."""

    def test_default_creation(self):
        """Test creating a GridBoard with default optional parameters."""
        gb = GridBoard(width=5, height=4, marker_size=0.04, marker_spacing=0.01)
        self.assertEqual(gb.width, 5)
        self.assertEqual(gb.height, 4)
        self.assertEqual(gb.marker_size, 0.04)
        self.assertEqual(gb.marker_spacing, 0.01)
        self.assertEqual(gb.dictionary_id, cv2.aruco.DICT_6X6_250)
        self.assertEqual(gb.border_bits, 1)
        self.assertFalse(gb.enable_symm_corners)
        self.assertTrue(gb.is_planar)
        self.assertEqual(gb.pattern_id, "grid_board")

    def test_custom_parameters(self):
        """Test creating a GridBoard with all custom parameters."""
        gb = GridBoard(
            width=3, height=2, marker_size=0.05, marker_spacing=0.015,
            dictionary_id=cv2.aruco.DICT_4X4_50, border_bits=2, is_planar=False
        )
        self.assertEqual(gb.width, 3)
        self.assertEqual(gb.height, 2)
        self.assertEqual(gb.marker_size, 0.05)
        self.assertEqual(gb.marker_spacing, 0.015)
        self.assertEqual(gb.dictionary_id, cv2.aruco.DICT_4X4_50)
        self.assertEqual(gb.border_bits, 2)
        self.assertFalse(gb.is_planar)

    def test_1x1_grid(self):
        """Test creating a minimal 1×1 grid board."""
        gb = GridBoard(width=1, height=1, marker_size=0.04, marker_spacing=0.01)
        self.assertEqual(gb.get_pattern_size(), (1, 1))

    def test_invalid_dimensions_zero(self):
        """Test that zero dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            GridBoard(width=0, height=4, marker_size=0.04, marker_spacing=0.01)

    def test_invalid_dimensions_negative(self):
        """Test that negative dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            GridBoard(width=-1, height=4, marker_size=0.04, marker_spacing=0.01)

    def test_invalid_marker_size(self):
        """Test that non-positive marker_size raises ValueError."""
        with self.assertRaises(ValueError):
            GridBoard(width=5, height=4, marker_size=0, marker_spacing=0.01)
        with self.assertRaises(ValueError):
            GridBoard(width=5, height=4, marker_size=-0.01, marker_spacing=0.01)

    def test_invalid_marker_spacing(self):
        """Test that non-positive marker_spacing raises ValueError."""
        with self.assertRaises(ValueError):
            GridBoard(width=5, height=4, marker_size=0.04, marker_spacing=0)

    def test_get_pattern_size(self):
        """Test get_pattern_size returns (width, height)."""
        gb = GridBoard(width=6, height=3, marker_size=0.04, marker_spacing=0.01)
        self.assertEqual(gb.get_pattern_size(), (6, 3))


class TestGridBoardSchema(unittest.TestCase):
    """Test GridBoard configuration schema."""

    def test_schema_structure(self):
        """Test that configuration schema has required fields."""
        schema = GridBoard.get_configuration_schema()
        self.assertIn("name", schema)
        self.assertIn("description", schema)
        self.assertIn("icon", schema)
        self.assertIn("parameters", schema)

    def test_schema_parameters(self):
        """Test that schema contains all expected parameters."""
        schema = GridBoard.get_configuration_schema()
        param_names = [p["name"] for p in schema["parameters"]]
        self.assertIn("width", param_names)
        self.assertIn("height", param_names)
        self.assertIn("marker_size", param_names)
        self.assertIn("marker_spacing", param_names)
        self.assertIn("border_bits", param_names)
        self.assertIn("enable_symm_corners", param_names)
        self.assertIn("dictionary_id", param_names)

    def test_schema_parameter_types(self):
        """Test that schema parameter types are correct."""
        schema = GridBoard.get_configuration_schema()
        params = {p["name"]: p for p in schema["parameters"]}
        self.assertEqual(params["width"]["type"], "integer")
        self.assertEqual(params["height"]["type"], "integer")
        self.assertEqual(params["marker_size"]["type"], "float")
        self.assertEqual(params["marker_spacing"]["type"], "float")
        self.assertEqual(params["border_bits"]["type"], "integer")
        self.assertEqual(params["dictionary_id"]["type"], "integer")

    def test_schema_border_bits_range(self):
        """Test that border_bits schema has correct min/max."""
        schema = GridBoard.get_configuration_schema()
        bb_param = next(p for p in schema["parameters"] if p["name"] == "border_bits")
        self.assertEqual(bb_param["default"], 1)
        self.assertEqual(bb_param["min"], 1)
        self.assertEqual(bb_param["max"], 5)

    def test_pattern_info(self):
        """Test PATTERN_INFO for auto-discovery."""
        info = GridBoard.PATTERN_INFO
        self.assertEqual(info["id"], "grid_board")
        self.assertIn("name", info)
        self.assertIn("icon", info)
        self.assertIn("category", info)

    def test_pattern_manager_discovery(self):
        """Test that GridBoard is discovered by the pattern manager."""
        manager = get_pattern_manager()
        patterns = manager.get_available_patterns()
        self.assertIn("grid_board", patterns)


class TestGridBoardSerialization(unittest.TestCase):
    """Test GridBoard JSON serialization and deserialization."""

    def test_to_json(self):
        """Test serialization to JSON."""
        gb = GridBoard(width=5, height=4, marker_size=0.04, marker_spacing=0.01, border_bits=2)
        data = gb.to_json()
        self.assertEqual(data["pattern_id"], "grid_board")
        self.assertTrue(data["is_planar"])
        params = data["parameters"]
        self.assertEqual(params["width"], 5)
        self.assertEqual(params["height"], 4)
        self.assertEqual(params["marker_size"], 0.04)
        self.assertEqual(params["marker_spacing"], 0.01)
        self.assertEqual(params["border_bits"], 2)
        self.assertFalse(params["enable_symm_corners"])
        self.assertEqual(params["total_markers"], 20)

    def test_from_json(self):
        """Test deserialization from JSON."""
        json_data = {
            "pattern_id": "grid_board",
            "parameters": {
                "width": 3, "height": 2,
                "marker_size": 0.05, "marker_spacing": 0.015,
                "dictionary_id": cv2.aruco.DICT_4X4_50,
                "border_bits": 3
            }
        }
        gb = GridBoard.from_json(json_data)
        self.assertEqual(gb.width, 3)
        self.assertEqual(gb.height, 2)
        self.assertEqual(gb.marker_size, 0.05)
        self.assertEqual(gb.marker_spacing, 0.015)
        self.assertEqual(gb.dictionary_id, cv2.aruco.DICT_4X4_50)
        self.assertEqual(gb.border_bits, 3)

    def test_from_json_defaults(self):
        """Test that from_json uses correct defaults for missing keys."""
        json_data = {"pattern_id": "grid_board", "parameters": {}}
        gb = GridBoard.from_json(json_data)
        self.assertEqual(gb.width, 5)
        self.assertEqual(gb.height, 7)
        self.assertEqual(gb.marker_size, 0.04)
        self.assertEqual(gb.marker_spacing, 0.01)
        self.assertEqual(gb.border_bits, 1)
        self.assertFalse(gb.enable_symm_corners)

    def test_roundtrip(self):
        """Test that serialize → deserialize preserves all parameters."""
        original = GridBoard(
            width=6, height=3, marker_size=0.035, marker_spacing=0.008,
            dictionary_id=cv2.aruco.DICT_5X5_100, border_bits=2
        )
        restored = GridBoard.from_json(original.to_json())
        self.assertEqual(restored.width, original.width)
        self.assertEqual(restored.height, original.height)
        self.assertEqual(restored.marker_size, original.marker_size)
        self.assertEqual(restored.marker_spacing, original.marker_spacing)
        self.assertEqual(restored.dictionary_id, original.dictionary_id)
        self.assertEqual(restored.border_bits, original.border_bits)
        self.assertEqual(restored.enable_symm_corners, original.enable_symm_corners)

    def test_json_string_roundtrip(self):
        """Test roundtrip through actual JSON string encoding."""
        gb = GridBoard(width=4, height=3, marker_size=0.04, marker_spacing=0.01, border_bits=2)
        json_str = json.dumps(gb.to_json())
        restored = GridBoard.from_json(json.loads(json_str))
        self.assertEqual(restored.width, 4)
        self.assertEqual(restored.border_bits, 2)

    def test_create_pattern_from_json(self):
        """Test creation through the module-level create_pattern_from_json."""
        json_data = {
            "pattern_id": "grid_board",
            "parameters": {
                "width": 5, "height": 4,
                "marker_size": 0.04, "marker_spacing": 0.01,
                "border_bits": 2
            }
        }
        pattern = create_pattern_from_json(json_data)
        self.assertIsInstance(pattern, GridBoard)
        self.assertEqual(pattern.border_bits, 2)

    def test_get_info(self):
        """Test get_info returns expected keys."""
        gb = GridBoard(width=5, height=4, marker_size=0.04, marker_spacing=0.01)
        info = gb.get_info()
        self.assertEqual(info["pattern_id"], "grid_board")
        self.assertIn("width", info)
        self.assertIn("height", info)
        self.assertIn("marker_size", info)
        self.assertIn("marker_spacing", info)
        self.assertIn("border_bits", info)
        self.assertIn("enable_symm_corners", info)

    def test_get_display_name(self):
        """Test get_display_name returns a readable string."""
        gb = GridBoard(width=5, height=4, marker_size=0.04, marker_spacing=0.01)
        name = gb.get_display_name()
        self.assertIn("5", name)
        self.assertIn("4", name)
        self.assertIn("40.0", name)


class TestGridBoardPatternImage(unittest.TestCase):
    """Test GridBoard pattern image generation."""

    def setUp(self):
        self.gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01)

    def test_generate_image_default(self):
        """Test default pattern image generation."""
        img = self.gb.generate_pattern_image()
        self.assertEqual(len(img.shape), 3)
        self.assertEqual(img.shape[2], 3)  # BGR
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)

    def test_generate_image_pixel_per_square(self):
        """Test pattern image generation with pixel_per_square."""
        img = self.gb.generate_pattern_image(pixel_per_square=100, border_pixels=20)
        self.assertEqual(len(img.shape), 3)

    def test_generate_image_pixels_per_meter(self):
        """Test pattern image generation with pixels_per_meter."""
        img = self.gb.generate_pattern_image(pixels_per_meter=1000, border_pixels=10)
        self.assertEqual(len(img.shape), 3)

    def test_border_bits_produce_different_images(self):
        """Test that different border_bits values produce different images."""
        gb1 = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01, border_bits=1)
        gb2 = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01, border_bits=2)
        img1 = gb1.generate_pattern_image(pixel_per_square=80, border_pixels=20)
        img2 = gb2.generate_pattern_image(pixel_per_square=80, border_pixels=20)
        # Same dimensions but different pixel content
        self.assertEqual(img1.shape, img2.shape)
        self.assertFalse(np.array_equal(img1, img2))


class TestGridBoardObjectPoints(unittest.TestCase):
    """Test GridBoard 3D object point generation."""

    def test_all_object_points_shape(self):
        """Test that all object points have correct shape."""
        gb = GridBoard(width=5, height=4, marker_size=0.04, marker_spacing=0.01)
        objp = gb.generate_object_points()
        # Shape depends on OpenCV version; verify 2D with 3 coords
        self.assertEqual(len(objp.shape), 2)
        self.assertEqual(objp.shape[1], 3)

    def test_object_points_planar(self):
        """Test that all z-coordinates are zero for planar board."""
        gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01)
        objp = gb.generate_object_points()
        np.testing.assert_array_equal(objp[:, 2], 0.0)

    def test_object_points_for_specific_ids(self):
        """Test object point generation for specific marker IDs."""
        gb = GridBoard(width=5, height=4, marker_size=0.04, marker_spacing=0.01)
        ids = np.array([0, 2, 5])
        objp = gb.generate_object_points(point_ids=ids)
        self.assertEqual(objp.shape, (3 * 4, 3))  # 3 markers × 4 corners

    def test_object_points_1x1(self):
        """Test object points for a 1×1 grid."""
        gb = GridBoard(width=1, height=1, marker_size=0.04, marker_spacing=0.01)
        objp = gb.generate_object_points()
        self.assertEqual(objp.shape, (4, 3))

    def test_manual_object_points_consistency(self):
        """Test that manual object points match expected positions."""
        gb = GridBoard(width=2, height=1, marker_size=0.04, marker_spacing=0.01)
        objp = gb._generate_manual_object_points()
        # First marker centered at (0, 0), second at (0.05, 0)
        half = 0.02
        step = 0.04 + 0.01  # marker_size + marker_spacing
        # Marker 0 corners
        np.testing.assert_allclose(objp[0], [-half, -half, 0], atol=1e-6)
        np.testing.assert_allclose(objp[1], [half, -half, 0], atol=1e-6)
        # Marker 1 top-left corner
        np.testing.assert_allclose(objp[4], [step - half, -half, 0], atol=1e-6)


class TestGridBoardDetection(unittest.TestCase):
    """Test GridBoard corner detection on generated images."""

    def test_detect_on_generated_image_all_supported_dictionaries(self):
        """Test detection succeeds for all OpenCV-supported ArUco dictionaries."""
        dict_items = _get_supported_aruco_dictionary_ids()
        self.assertGreater(len(dict_items), 0, "No supported ArUco dictionaries found")

        for dict_id, dict_name in dict_items:
            with self.subTest(dictionary_id=dict_id, dictionary_name=dict_name):
                gb = GridBoard(
                    width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                    dictionary_id=dict_id
                )
                img = gb.generate_pattern_image(pixel_per_square=100, border_pixels=50)
                success, corners, ids = gb.detect_corners(img)

                self.assertTrue(success, f"Detection failed for dictionary {dict_name} ({dict_id})")
                self.assertIsNotNone(corners)
                self.assertIsNotNone(ids)
                self.assertEqual(len(ids), 6)
                self.assertEqual(corners.shape, (24, 2))

    def test_detect_on_generated_image(self):
        """Test that markers can be detected on a generated grid board image."""
        gb = GridBoard(
            width=3, height=2, marker_size=0.04, marker_spacing=0.01,
            dictionary_id=cv2.aruco.DICT_4X4_50
        )
        img = gb.generate_pattern_image(pixel_per_square=100, border_pixels=50)
        success, corners, ids = gb.detect_corners(img)
        self.assertTrue(success)
        self.assertIsNotNone(corners)
        self.assertIsNotNone(ids)
        # Should detect all 6 markers
        self.assertEqual(len(ids), 6)
        # 6 markers × 4 corners = 24 points
        self.assertEqual(corners.shape[0], 24)
        self.assertEqual(corners.shape[1], 2)

    def test_detect_ids_sorted(self):
        """Test that detected marker IDs are sorted."""
        gb = GridBoard(
            width=3, height=2, marker_size=0.04, marker_spacing=0.01,
            dictionary_id=cv2.aruco.DICT_4X4_50
        )
        img = gb.generate_pattern_image(pixel_per_square=100, border_pixels=50)
        success, corners, ids = gb.detect_corners(img)
        self.assertTrue(success)
        self.assertTrue(np.all(ids[:-1] <= ids[1:]))

    def test_detect_on_blank_image(self):
        """Test detection on a blank white image returns failure."""
        gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01)
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 255
        success, corners, ids = gb.detect_corners(blank)
        self.assertFalse(success)
        self.assertIsNone(corners)
        self.assertIsNone(ids)

    def test_detect_with_border_bits_and_symm_corners(self):
        """Test detection with all border_bits and symm_corners combinations."""
        for bb in [1, 2, 3]:
            for sc in [False, True]:
                with self.subTest(border_bits=bb, enable_symm_corners=sc):
                    gb = GridBoard(
                        width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                        dictionary_id=cv2.aruco.DICT_4X4_50,
                        border_bits=bb, enable_symm_corners=sc
                    )
                    img = gb.generate_pattern_image(pixel_per_square=120, border_pixels=50)
                    success, corners, ids = gb.detect_corners(img)
                    self.assertTrue(success,
                        f"Detection failed with border_bits={bb}, symm_corners={sc}")
                    self.assertEqual(len(ids), 6)

    def test_detect_all_combinations(self):
        """Test detection with all combinations of border_bits, symm_corners, reverse_x, reverse_y."""
        for bb in [1, 2]:
            for sc in [False, True]:
                for rx in [False, True]:
                    for ry in [False, True]:
                        with self.subTest(border_bits=bb, symm=sc, rev_x=rx, rev_y=ry):
                            gb = GridBoard(
                                width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                                dictionary_id=cv2.aruco.DICT_4X4_50,
                                border_bits=bb, enable_symm_corners=sc,
                                reverse_x=rx, reverse_y=ry
                            )
                            img = gb.generate_pattern_image(pixel_per_square=120, border_pixels=50)
                            success, corners, ids = gb.detect_corners(img)
                            self.assertTrue(success,
                                f"Detection failed: bb={bb}, symm={sc}, rx={rx}, ry={ry}")
                            self.assertEqual(len(ids), 6,
                                f"Expected 6 markers: bb={bb}, symm={sc}, rx={rx}, ry={ry}")


class TestGridBoardDrawCorners(unittest.TestCase):
    """Test GridBoard corner drawing."""

    def test_draw_corners_returns_image(self):
        """Test that draw_corners returns a valid image."""
        gb = GridBoard(
            width=3, height=2, marker_size=0.04, marker_spacing=0.01,
            dictionary_id=cv2.aruco.DICT_4X4_50
        )
        img = gb.generate_pattern_image(pixel_per_square=100, border_pixels=50)
        success, corners, ids = gb.detect_corners(img)
        self.assertTrue(success)
        drawn = gb.draw_corners(img, corners, ids)
        self.assertEqual(drawn.shape, img.shape)

    def test_draw_corners_does_not_modify_original(self):
        """Test that draw_corners does not modify the original image."""
        gb = GridBoard(
            width=3, height=2, marker_size=0.04, marker_spacing=0.01,
            dictionary_id=cv2.aruco.DICT_4X4_50
        )
        img = gb.generate_pattern_image(pixel_per_square=100, border_pixels=50)
        original = img.copy()
        success, corners, ids = gb.detect_corners(img)
        self.assertTrue(success)
        gb.draw_corners(img, corners, ids)
        np.testing.assert_array_equal(img, original)

    def test_draw_corners_no_corners(self):
        """Test draw_corners with None corners returns original image."""
        gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = gb.draw_corners(img, None)
        np.testing.assert_array_equal(result, img)


class TestGridBoardSymmCorners(unittest.TestCase):
    """Test GridBoard symmetric corners (AprilTag/Kalibr style)."""

    def test_enable_symm_corners_creation(self):
        """Test creating a GridBoard with enable_symm_corners=True."""
        gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                       enable_symm_corners=True)
        self.assertTrue(gb.enable_symm_corners)

    def test_symm_corners_json_roundtrip(self):
        """Test that enable_symm_corners survives JSON roundtrip."""
        original = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                             enable_symm_corners=True)
        restored = GridBoard.from_json(original.to_json())
        self.assertTrue(restored.enable_symm_corners)

    def test_symm_corners_from_json_true(self):
        """Test from_json with enable_symm_corners=True."""
        json_data = {
            "pattern_id": "grid_board",
            "parameters": {
                "width": 3, "height": 2,
                "marker_size": 0.04, "marker_spacing": 0.01,
                "enable_symm_corners": True
            }
        }
        gb = GridBoard.from_json(json_data)
        self.assertTrue(gb.enable_symm_corners)

    def test_corner_gaps_filled_black(self):
        """Test that all corner gap cells are black when enable_symm_corners=True."""
        pps = 100  # pixels per square (marker)
        border = 30
        gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                       enable_symm_corners=True)
        img = gb.generate_pattern_image(pixel_per_square=pps, border_pixels=border)
        spacing_px = int(0.01 * int(pps / 0.04))
        step = pps + spacing_px
        # All (W+1)*(H+1) vertex gap cells should be black
        for gj in range(3):  # height + 1
            for gi in range(4):  # width + 1
                gx = border + gi * step - spacing_px
                gy = border + gj * step - spacing_px
                x0, y0 = max(0, gx), max(0, gy)
                x1 = min(img.shape[1], gx + spacing_px)
                y1 = min(img.shape[0], gy + spacing_px)
                cell = img[y0:y1, x0:x1]
                self.assertTrue(np.all(cell == 0),
                    f"Corner gap ({gi},{gj}) should be black")

    def test_corner_gaps_white_when_disabled(self):
        """Test that corner gap cells are white when enable_symm_corners=False."""
        pps = 100
        border = 20
        gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                       enable_symm_corners=False)
        img = gb.generate_pattern_image(pixel_per_square=pps, border_pixels=border)
        spacing_px = int(0.01 * int(pps / 0.04))
        for j in range(1):
            for i in range(2):
                gx = border + i * (pps + spacing_px) + pps
                gy = border + j * (pps + spacing_px) + pps
                cell = img[gy:gy + spacing_px, gx:gx + spacing_px]
                self.assertTrue(np.all(cell == 255),
                    f"Corner gap ({i},{j}) should be white when disabled")

    def test_symm_corners_1x1_has_corner_squares(self):
        """Test that a 1x1 grid with symm corners gets 4 corner squares."""
        gb_on = GridBoard(width=1, height=1, marker_size=0.04, marker_spacing=0.01,
                          enable_symm_corners=True)
        gb_off = GridBoard(width=1, height=1, marker_size=0.04, marker_spacing=0.01,
                           enable_symm_corners=False)
        img_on = gb_on.generate_pattern_image(pixel_per_square=80, border_pixels=30)
        img_off = gb_off.generate_pattern_image(pixel_per_square=80, border_pixels=30)
        # With boundary corners, symm_corners should differ even for 1x1
        self.assertFalse(np.array_equal(img_on, img_off))

    def test_symm_corners_differ_from_standard(self):
        """Test that symm_corners image differs from standard image."""
        kwargs = dict(width=3, height=2, marker_size=0.04, marker_spacing=0.01)
        gb_on = GridBoard(enable_symm_corners=True, **kwargs)
        gb_off = GridBoard(enable_symm_corners=False, **kwargs)
        img_on = gb_on.generate_pattern_image(pixel_per_square=80, border_pixels=20)
        img_off = gb_off.generate_pattern_image(pixel_per_square=80, border_pixels=20)
        self.assertEqual(img_on.shape, img_off.shape)
        self.assertFalse(np.array_equal(img_on, img_off))


class TestGridBoardReverseXY(unittest.TestCase):
    """Test GridBoard reverse_x and reverse_y parameters."""

    def test_default_no_reverse(self):
        """Test that defaults are False."""
        gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01)
        self.assertFalse(gb.reverse_x)
        self.assertFalse(gb.reverse_y)

    def test_reverse_x_creation(self):
        """Test creating with reverse_x=True."""
        gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                       reverse_x=True)
        self.assertTrue(gb.reverse_x)
        self.assertFalse(gb.reverse_y)

    def test_reverse_y_creation(self):
        """Test creating with reverse_y=True."""
        gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                       reverse_y=True)
        self.assertFalse(gb.reverse_x)
        self.assertTrue(gb.reverse_y)

    def test_reverse_both_creation(self):
        """Test creating with both reverse_x and reverse_y True."""
        gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                       reverse_x=True, reverse_y=True)
        self.assertTrue(gb.reverse_x)
        self.assertTrue(gb.reverse_y)

    def test_json_roundtrip(self):
        """Test that reverse_x/reverse_y survive JSON roundtrip."""
        original = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                             reverse_x=True, reverse_y=True)
        restored = GridBoard.from_json(original.to_json())
        self.assertTrue(restored.reverse_x)
        self.assertTrue(restored.reverse_y)

    def test_from_json_defaults(self):
        """Test that from_json defaults reverse_x/reverse_y to False."""
        json_data = {"pattern_id": "grid_board", "parameters": {}}
        gb = GridBoard.from_json(json_data)
        self.assertFalse(gb.reverse_x)
        self.assertFalse(gb.reverse_y)

    def test_to_json_contains_reverse(self):
        """Test that to_json includes reverse_x and reverse_y."""
        gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                       reverse_x=True)
        data = gb.to_json()
        self.assertTrue(data["parameters"]["reverse_x"])
        self.assertFalse(data["parameters"]["reverse_y"])

    def test_reverse_x_marker_layout(self):
        """Test that reverse_x puts last column ID at left side of image."""
        gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                       dictionary_id=cv2.aruco.DICT_4X4_50, reverse_x=True)
        img = gb.generate_pattern_image(pixel_per_square=100, border_pixels=50)
        success, corners, ids = gb.detect_corners(img)
        self.assertTrue(success)
        self.assertEqual(len(ids), 6)
        # Group corners by marker ID and find centers
        centers = {}
        for i, mid in enumerate(ids):
            marker_corners = corners[i*4:(i+1)*4]
            centers[mid] = marker_corners.mean(axis=0)
        # ID 2 (rightmost in normal, col=2) should be at left (smallest x)
        # ID 0 (leftmost in normal, col=0) should be at right (largest x)
        self.assertLess(centers[2][0], centers[0][0])

    def test_reverse_y_marker_layout(self):
        """Test that reverse_y puts last row ID at top of image."""
        gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                       dictionary_id=cv2.aruco.DICT_4X4_50, reverse_y=True)
        img = gb.generate_pattern_image(pixel_per_square=100, border_pixels=50)
        success, corners, ids = gb.detect_corners(img)
        self.assertTrue(success)
        centers = {}
        for i, mid in enumerate(ids):
            centers[mid] = corners[i*4:(i+1)*4].mean(axis=0)
        # ID 3 (row 1 in normal) should be at top (smallest y) in reversed image
        # ID 0 (row 0 in normal) should be at bottom (largest y) in reversed image
        self.assertLess(centers[3][1], centers[0][1])

    def test_reverse_both_marker_layout(self):
        """Test that reverse_x + reverse_y puts ID W*H-1 at top-left."""
        gb = GridBoard(width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                       dictionary_id=cv2.aruco.DICT_4X4_50,
                       reverse_x=True, reverse_y=True)
        img = gb.generate_pattern_image(pixel_per_square=100, border_pixels=50)
        success, corners, ids = gb.detect_corners(img)
        self.assertTrue(success)
        centers = {}
        for i, mid in enumerate(ids):
            centers[mid] = corners[i*4:(i+1)*4].mean(axis=0)
        # ID 5 (last marker) should be at top-left (smallest x and y)
        # ID 0 (first marker) should be at bottom-right (largest x and y)
        self.assertLess(centers[5][0], centers[0][0])
        self.assertLess(centers[5][1], centers[0][1])

    def test_reverse_x_object_points(self):
        """Test that object points reflect the reversed x positions."""
        gb_normal = GridBoard(width=3, height=1, marker_size=0.04, marker_spacing=0.01)
        gb_rev = GridBoard(width=3, height=1, marker_size=0.04, marker_spacing=0.01,
                           reverse_x=True)
        objp_normal = gb_normal.generate_object_points()
        objp_rev = gb_rev.generate_object_points()
        # In reversed x: ID 0 should be at the position where ID 2 was in normal
        # ID 0 corners in normal are at x ~ 0, in reversed at x ~ 0.1
        self.assertGreater(objp_rev[0, 0], objp_normal[0, 0])

    def test_reverse_y_object_points(self):
        """Test that object points reflect the reversed y positions."""
        gb_normal = GridBoard(width=1, height=2, marker_size=0.04, marker_spacing=0.01)
        gb_rev = GridBoard(width=1, height=2, marker_size=0.04, marker_spacing=0.01,
                           reverse_y=True)
        objp_normal = gb_normal.generate_object_points()
        objp_rev = gb_rev.generate_object_points()
        # In reversed y: ID 0 should be at the position where ID 1 was in normal
        self.assertGreater(objp_rev[0, 1], objp_normal[0, 1])

    def test_detect_with_reverse_and_symm_corners(self):
        """Test detection works with reverse + symm_corners combinations."""
        for rx, ry in [(True, False), (False, True), (True, True)]:
            for sc in [False, True]:
                with self.subTest(reverse_x=rx, reverse_y=ry, symm_corners=sc):
                    gb = GridBoard(
                        width=3, height=2, marker_size=0.04, marker_spacing=0.01,
                        dictionary_id=cv2.aruco.DICT_4X4_50,
                        reverse_x=rx, reverse_y=ry, enable_symm_corners=sc
                    )
                    img = gb.generate_pattern_image(pixel_per_square=120, border_pixels=50)
                    success, corners, ids = gb.detect_corners(img)
                    self.assertTrue(success,
                        f"Detection failed: reverse_x={rx}, reverse_y={ry}, symm={sc}")
                    self.assertEqual(len(ids), 6)

    def test_schema_has_reverse_params(self):
        """Test that schema includes reverse_x and reverse_y."""
        schema = GridBoard.get_configuration_schema()
        param_names = [p["name"] for p in schema["parameters"]]
        self.assertIn("reverse_x", param_names)
        self.assertIn("reverse_y", param_names)


if __name__ == '__main__':
    unittest.main()
