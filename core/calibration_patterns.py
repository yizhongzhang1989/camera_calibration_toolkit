"""
Calibration Pattern System - Legacy Compatibility Layer
======================================================

This module provides backward compatibility for the original calibration patterns API
while using the new modular pattern system internally.

The actual pattern implementations have been moved to the calibration_patterns/
directory for better modularity and auto-discovery.

Legacy API preserved:
- CalibrationPattern, StandardChessboard, CharucoBoard classes
- CalibrationPatternManager class  
- create_chessboard_pattern() function
- get_pattern_type_configurations() function
- create_pattern_from_json() function
- COMMON_PATTERNS and get_common_pattern() functions

New modular system available at:
- from core.calibration_patterns import get_pattern_manager
"""

# Import all classes and functions from the new modular system
from .calibration_patterns import *
from .calibration_patterns.base import CalibrationPattern
from .calibration_patterns.standard_chessboard import StandardChessboard
from .calibration_patterns.charuco_board import CharucoBoard
from .calibration_patterns.manager import CalibrationPatternManager, get_pattern_manager

# Legacy compatibility exports
__all__ = [
    # Base classes
    'CalibrationPattern',
    'StandardChessboard', 
    'CharucoBoard',
    'CalibrationPatternManager',
    
    # Functions
    'get_pattern_manager',
    'get_pattern_type_configurations',
    'create_pattern_from_json',
    'create_chessboard_pattern',
    'get_common_pattern',
    
    # Constants
    'COMMON_PATTERNS'
]

# Legacy COMMON_PATTERNS constant for backward compatibility
COMMON_PATTERNS = {
    'standard_8x6': {
        'pattern_type': 'standard',
        'width': 8,
        'height': 6, 
        'square_size': 0.025
    },
    'standard_9x6': {
        'pattern_type': 'standard',
        'width': 9,
        'height': 6,
        'square_size': 0.025
    },
    'standard_11x8': {
        'pattern_type': 'standard',
        'width': 11,
        'height': 8,
        'square_size': 0.025
    },
    'charuco_8x6': {
        'pattern_type': 'charuco',
        'width': 8,
        'height': 6,
        'square_size': 0.040,
        'marker_size': 0.020
    },
    'charuco_9x7': {
        'pattern_type': 'charuco',
        'width': 9,
        'height': 7,
        'square_size': 0.025,
        'marker_size': 0.015
    }
}

def get_common_pattern(pattern_name: str):
    """
    Get a common pattern by name (legacy compatibility function).
    
    Args:
        pattern_name: Name of the common pattern
        
    Returns:
        CalibrationPattern instance
    """
    if pattern_name not in COMMON_PATTERNS:
        raise ValueError(f"Unknown common pattern: {pattern_name}. "
                        f"Available patterns: {list(COMMON_PATTERNS.keys())}")
    
    config = COMMON_PATTERNS[pattern_name]
    pattern_type = config.pop('pattern_type')
    return create_chessboard_pattern(pattern_type, **config)
