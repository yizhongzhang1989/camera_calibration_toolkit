#!/usr/bin/env python3
"""
Unit Tests for Pattern Serialization
====================================

This module contains comprehensive unit tests for the pattern serialization functionality.

Test Classes:
1. TestPatternSerialization:
   - Tests JSON serialization and deserialization of all pattern types
   - Validates that serialized patterns can be correctly restored
   - Ensures parameter preservation across serialization/deserialization cycles
   - Tests error handling for invalid JSON data

The pattern serialization system is used in:
- Calibration report generation
- Pattern configuration storage
- Calibration result archiving
- Pattern recreation for analysis
"""

import unittest
import os
import sys
import json
import cv2
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.calibration_patterns import (
    get_pattern_manager,
    load_pattern_from_json,
    save_pattern_to_json
)


class TestPatternSerialization(unittest.TestCase):
    """Test pattern serialization and deserialization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = get_pattern_manager()

    def test_standard_chessboard_serialization(self):
        """Test serialization of standard chessboard patterns."""
        # Create a standard chessboard pattern
        original_pattern = self.manager.create_pattern(
            'standard_chessboard',
            width=11,
            height=8,
            square_size=0.025
        )

        # Serialize to JSON
        pattern_dict = original_pattern.to_json()
        
        # Verify JSON structure
        self.assertIn('pattern_id', pattern_dict)
        self.assertIn('name', pattern_dict)
        self.assertIn('description', pattern_dict)
        self.assertIn('is_planar', pattern_dict)
        self.assertIn('parameters', pattern_dict)
        
        self.assertEqual(pattern_dict['pattern_id'], 'standard_chessboard')
        self.assertTrue(pattern_dict['is_planar'])
        
        # Verify parameters
        params = pattern_dict['parameters']
        self.assertEqual(params['width'], 11)
        self.assertEqual(params['height'], 8)
        self.assertEqual(params['square_size'], 0.025)
        self.assertIn('total_corners', params)
        self.assertEqual(params['total_corners'], 11 * 8)

        # Deserialize from JSON
        restored_pattern = load_pattern_from_json(pattern_dict)
        
        # Verify restored pattern
        self.assertEqual(restored_pattern.pattern_id, original_pattern.pattern_id)
        self.assertEqual(restored_pattern.name, original_pattern.name)
        self.assertEqual(restored_pattern.width, original_pattern.width)
        self.assertEqual(restored_pattern.height, original_pattern.height)
        self.assertEqual(restored_pattern.square_size, original_pattern.square_size)

    def test_charuco_board_serialization(self):
        """Test serialization of ChArUco board patterns."""
        # Create a ChArUco board pattern
        original_pattern = self.manager.create_pattern(
            'charuco_board',
            width=8,
            height=6,
            square_size=0.03,
            marker_size=0.02,
            dictionary_id=cv2.aruco.DICT_4X4_100
        )

        # Serialize to JSON
        pattern_dict = original_pattern.to_json()
        
        # Verify JSON structure
        self.assertEqual(pattern_dict['pattern_id'], 'charuco_board')
        self.assertTrue(pattern_dict['is_planar'])
        
        # Verify parameters
        params = pattern_dict['parameters']
        self.assertEqual(params['width'], 8)
        self.assertEqual(params['height'], 6)
        self.assertEqual(params['square_size'], 0.03)
        self.assertEqual(params['marker_size'], 0.02)
        self.assertEqual(params['dictionary_id'], cv2.aruco.DICT_4X4_100)
        self.assertIn('total_corners', params)

        # Deserialize from JSON
        restored_pattern = load_pattern_from_json(pattern_dict)
        
        # Verify restored pattern
        self.assertEqual(restored_pattern.pattern_id, original_pattern.pattern_id)
        self.assertEqual(restored_pattern.width, original_pattern.width)
        self.assertEqual(restored_pattern.height, original_pattern.height)
        self.assertEqual(restored_pattern.square_size, original_pattern.square_size)
        self.assertEqual(restored_pattern.marker_size, original_pattern.marker_size)
        self.assertEqual(restored_pattern.dictionary_id, original_pattern.dictionary_id)

    def test_grid_board_serialization(self):
        """Test serialization of ArUco grid board patterns."""
        # Create a grid board pattern
        original_pattern = self.manager.create_pattern(
            'grid_board',
            markers_x=5,
            markers_y=4,
            marker_size=0.04,
            marker_separation=0.01,
            dictionary_id=cv2.aruco.DICT_6X6_250
        )

        # Serialize to JSON
        pattern_dict = original_pattern.to_json()
        
        # Verify JSON structure
        self.assertEqual(pattern_dict['pattern_id'], 'grid_board')
        self.assertTrue(pattern_dict['is_planar'])
        
        # Verify parameters
        params = pattern_dict['parameters']
        self.assertEqual(params['markers_x'], 5)
        self.assertEqual(params['markers_y'], 4)
        self.assertEqual(params['marker_size'], 0.04)
        self.assertEqual(params['marker_separation'], 0.01)
        self.assertEqual(params['dictionary_id'], cv2.aruco.DICT_6X6_250)
        self.assertIn('total_markers', params)
        self.assertEqual(params['total_markers'], 5 * 4)

        # Deserialize from JSON
        restored_pattern = load_pattern_from_json(pattern_dict)
        
        # Verify restored pattern
        self.assertEqual(restored_pattern.pattern_id, original_pattern.pattern_id)
        self.assertEqual(restored_pattern.markers_x, original_pattern.markers_x)
        self.assertEqual(restored_pattern.markers_y, original_pattern.markers_y)
        self.assertEqual(restored_pattern.marker_size, original_pattern.marker_size)
        self.assertEqual(restored_pattern.marker_separation, original_pattern.marker_separation)
        self.assertEqual(restored_pattern.dictionary_id, original_pattern.dictionary_id)

    def test_json_string_serialization(self):
        """Test serialization using JSON string format."""
        # Create pattern
        original_pattern = self.manager.create_pattern(
            'standard_chessboard',
            width=9,
            height=6,
            square_size=0.02
        )

        # Serialize using save_pattern_to_json function
        pattern_dict = save_pattern_to_json(original_pattern)
        
        # Convert to JSON string and back
        json_string = json.dumps(pattern_dict, indent=2)
        parsed_dict = json.loads(json_string)
        
        # Deserialize from parsed dictionary
        restored_pattern = load_pattern_from_json(parsed_dict)
        
        # Verify patterns match
        self.assertEqual(restored_pattern.pattern_id, original_pattern.pattern_id)
        self.assertEqual(restored_pattern.width, original_pattern.width)
        self.assertEqual(restored_pattern.height, original_pattern.height)
        self.assertEqual(restored_pattern.square_size, original_pattern.square_size)

    def test_serialization_roundtrip_all_patterns(self):
        """Test serialization roundtrip for all pattern types."""
        test_patterns = [
            ('standard_chessboard', {
                'width': 7, 'height': 5, 'square_size': 0.03
            }),
            ('charuco_board', {
                'width': 6, 'height': 4, 'square_size': 0.025,
                'marker_size': 0.02, 'dictionary_id': cv2.aruco.DICT_5X5_250
            }),
            ('grid_board', {
                'markers_x': 3, 'markers_y': 2, 'marker_size': 0.05,
                'marker_separation': 0.015, 'dictionary_id': cv2.aruco.DICT_4X4_50
            })
        ]

        for pattern_id, params in test_patterns:
            with self.subTest(pattern_id=pattern_id):
                # Create original pattern
                original_pattern = self.manager.create_pattern(pattern_id, **params)
                
                # Serialize and deserialize
                pattern_dict = original_pattern.to_json()
                restored_pattern = load_pattern_from_json(pattern_dict)
                
                # Verify basic properties match
                self.assertEqual(restored_pattern.pattern_id, original_pattern.pattern_id)
                self.assertEqual(restored_pattern.name, original_pattern.name)
                self.assertEqual(restored_pattern.is_planar, original_pattern.is_planar)
                
                # Verify parameter-specific properties
                if pattern_id in ['standard_chessboard', 'charuco_board']:
                    self.assertEqual(restored_pattern.width, original_pattern.width)
                    self.assertEqual(restored_pattern.height, original_pattern.height)
                    self.assertEqual(restored_pattern.square_size, original_pattern.square_size)
                
                if pattern_id == 'charuco_board':
                    self.assertEqual(restored_pattern.marker_size, original_pattern.marker_size)
                    self.assertEqual(restored_pattern.dictionary_id, original_pattern.dictionary_id)
                
                if pattern_id == 'grid_board':
                    self.assertEqual(restored_pattern.markers_x, original_pattern.markers_x)
                    self.assertEqual(restored_pattern.markers_y, original_pattern.markers_y)
                    self.assertEqual(restored_pattern.marker_size, original_pattern.marker_size)
                    self.assertEqual(restored_pattern.marker_separation, original_pattern.marker_separation)
                    self.assertEqual(restored_pattern.dictionary_id, original_pattern.dictionary_id)

    def test_invalid_json_handling(self):
        """Test error handling for invalid JSON data."""
        # Test missing pattern_id
        invalid_json = {
            'name': 'Test Pattern',
            'description': 'Test',
            'is_planar': True,
            'parameters': {'width': 8, 'height': 6}
        }
        
        with self.assertRaises(KeyError):
            load_pattern_from_json(invalid_json)

        # Test invalid pattern_id
        invalid_json = {
            'pattern_id': 'nonexistent_pattern',
            'name': 'Test Pattern',
            'description': 'Test',
            'is_planar': True,
            'parameters': {'width': 8, 'height': 6}
        }
        
        with self.assertRaises(KeyError):
            load_pattern_from_json(invalid_json)

        # Test invalid parameter types (this should work with robustness in mind)
        # The pattern classes use default values for missing parameters
        valid_but_incomplete_json = {
            'pattern_id': 'standard_chessboard',
            'name': 'Test Pattern',
            'description': 'Test',
            'is_planar': True,
            'parameters': {'width': 8}  # Missing height and square_size - should use defaults
        }
        
        # This should succeed due to default value handling
        pattern = load_pattern_from_json(valid_but_incomplete_json)
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.width, 8)
        self.assertEqual(pattern.height, 8)  # Default value
        self.assertEqual(pattern.square_size, 0.025)  # Default value

    def test_parameter_validation(self):
        """Test that serialized parameters are validated correctly."""
        # Create pattern with specific parameters
        original_pattern = self.manager.create_pattern(
            'charuco_board',
            width=10,
            height=7,
            square_size=0.035,
            marker_size=0.025,
            dictionary_id=cv2.aruco.DICT_7X7_1000
        )

        # Serialize
        pattern_dict = original_pattern.to_json()
        
        # Verify all expected parameters are present
        params = pattern_dict['parameters']
        required_params = ['width', 'height', 'square_size', 'marker_size', 'dictionary_id', 'total_corners']
        
        for param in required_params:
            self.assertIn(param, params, f"Missing parameter: {param}")
        
        # Verify parameter types
        self.assertIsInstance(params['width'], int)
        self.assertIsInstance(params['height'], int)
        self.assertIsInstance(params['square_size'], (int, float))
        self.assertIsInstance(params['marker_size'], (int, float))
        self.assertIsInstance(params['dictionary_id'], int)
        self.assertIsInstance(params['total_corners'], int)

    def test_json_file_operations(self):
        """Test saving and loading patterns from JSON files."""
        import tempfile
        
        # Create a test pattern
        original_pattern = self.manager.create_pattern(
            'standard_chessboard',
            width=12,
            height=9,
            square_size=0.022
        )

        # Test saving to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            pattern_dict = original_pattern.to_json()
            json.dump(pattern_dict, f, indent=2)

        try:
            # Test loading from file
            with open(temp_file, 'r') as f:
                loaded_dict = json.load(f)
            
            restored_pattern = load_pattern_from_json(loaded_dict)
            
            # Verify patterns match
            self.assertEqual(restored_pattern.pattern_id, original_pattern.pattern_id)
            self.assertEqual(restored_pattern.width, original_pattern.width)
            self.assertEqual(restored_pattern.height, original_pattern.height)
            self.assertEqual(restored_pattern.square_size, original_pattern.square_size)
            
        finally:
            # Clean up
            os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()
