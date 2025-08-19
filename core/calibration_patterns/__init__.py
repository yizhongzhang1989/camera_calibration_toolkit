"""
Modular Calibration Pattern System
=================================

This package provides a modular, auto-discoverable calibration pattern system
for the Camera Calibration Toolkit.

Key Features:
- Automatic pattern discovery and registration
- Modular pattern definitions (one file per pattern)
- Dynamic UI generation from pattern schemas
- Backward compatibility with existing APIs

Usage:
    from core.calibration_patterns import get_pattern_manager
    
    manager = get_pattern_manager()
    configurations = manager.get_pattern_configurations()
    pattern = manager.create_pattern('standard_chessboard', width=11, height=8, square_size=0.025)
"""

# Import base classes
from .base import CalibrationPattern
from .manager import CalibrationPatternManager, get_pattern_manager

# Import concrete pattern classes for backward compatibility
from .standard_chessboard import StandardChessboard
from .charuco_board import CharucoBoard

# Backward compatibility exports - these maintain the existing API
from .manager import get_pattern_manager

def get_pattern_type_configurations():
    """Get pattern type configurations for the web API (backward compatibility)."""
    manager = get_pattern_manager()
    return manager.get_pattern_configurations()

def create_pattern_from_json(json_data):
    """Create pattern from JSON data (backward compatibility)."""
    manager = get_pattern_manager()
    pattern_id = json_data.get('pattern_id', '')
    
    # Handle legacy pattern IDs
    if pattern_id == 'standard':
        pattern_id = 'standard_chessboard'
    elif pattern_id == 'charuco':
        pattern_id = 'charuco_board'
    
    # Extract parameters from JSON
    parameters = json_data.get('parameters', {})
    
    return manager.create_pattern(pattern_id, **parameters)

# Legacy function names for backward compatibility
def create_chessboard_pattern(pattern_type, **kwargs):
    """Create chessboard pattern (backward compatibility)."""
    manager = get_pattern_manager()
    
    # Map legacy pattern types to new IDs
    pattern_id_map = {
        'standard': 'standard_chessboard',
        'charuco': 'charuco_board'
    }
    
    pattern_id = pattern_id_map.get(pattern_type, pattern_type)
    return manager.create_pattern(pattern_id, **kwargs)

# Legacy COMMON_PATTERNS constant for backward compatibility
COMMON_PATTERNS = {
    'standard_8x6': {
        'pattern_type': 'standard',
        'width': 8,
        'height': 6, 
        'square_size': 0.025,
        'description': 'Standard 8x6 chessboard (25mm squares)'
    },
    'standard_9x6': {
        'pattern_type': 'standard',
        'width': 9,
        'height': 6,
        'square_size': 0.025,
        'description': 'Standard 9x6 chessboard (25mm squares)'
    },
    'standard_11x8': {
        'pattern_type': 'standard',
        'width': 11,
        'height': 8,
        'square_size': 0.025,
        'description': 'Standard 11x8 chessboard (25mm squares)'
    },
    'charuco_8x6': {
        'pattern_type': 'charuco',
        'width': 8,
        'height': 6,
        'square_size': 0.040,
        'marker_size': 0.020,
        'description': 'ChArUco 8x6 board (40mm squares, 20mm markers)'
    },
    'charuco_9x7': {
        'pattern_type': 'charuco',
        'width': 9,
        'height': 7,
        'square_size': 0.025,
        'marker_size': 0.015,
        'description': 'ChArUco 9x7 board (25mm squares, 15mm markers)'
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
    
    config = COMMON_PATTERNS[pattern_name].copy()  # Copy to avoid modifying original
    pattern_type = config.pop('pattern_type')
    config.pop('description', None)  # Remove description if present
    return create_chessboard_pattern(pattern_type, **config)

# Export key classes and functions
__all__ = [
    'CalibrationPattern',
    'StandardChessboard',
    'CharucoBoard',
    'CalibrationPatternManager', 
    'get_pattern_manager',
    'get_pattern_type_configurations',
    'create_pattern_from_json',
    'create_chessboard_pattern',
    'get_common_pattern',
    'COMMON_PATTERNS'
]
