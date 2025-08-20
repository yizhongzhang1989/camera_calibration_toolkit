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
from .grid_board import GridBoard

def get_pattern_type_configurations():
    """Get pattern type configurations for the web API (backward compatibility).
    
    Note: This is a legacy function. New code should use:
    manager = get_pattern_manager()
    configurations = manager.get_pattern_configurations()
    """
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
    elif pattern_id == 'grid':
        pattern_id = 'grid_board'
    
    # Extract parameters from JSON
    parameters = json_data.get('parameters', {})
    
    return manager.create_pattern(pattern_id, **parameters)

# Legacy function names for backward compatibility
def create_chessboard_pattern(pattern_type, **kwargs):
    """Create chessboard pattern (backward compatibility).
    
    Note: This is a legacy function. New code should use:
    manager = get_pattern_manager()
    pattern = manager.create_pattern(pattern_id, **parameters)
    """
    manager = get_pattern_manager()
    
    # Map legacy pattern types to new IDs
    pattern_id_map = {
        'standard': 'standard_chessboard',
        'charuco': 'charuco_board'
    }
    
    pattern_id = pattern_id_map.get(pattern_type, pattern_type)
    return manager.create_pattern(pattern_id, **kwargs)

# Export key classes and functions
__all__ = [
    'CalibrationPattern',
    'StandardChessboard',
    'CharucoBoard',
    'GridBoard',
    'CalibrationPatternManager', 
    'get_pattern_manager',
    'get_pattern_type_configurations',
    'create_pattern_from_json',
    'create_chessboard_pattern'
]
