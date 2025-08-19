"""
Pattern Manager for Auto-Discovery System
========================================

This module manages the automatic discovery and registration of 
calibration pattern types in the modular pattern system.
"""

import os
import importlib
import inspect
from typing import Dict, Type, Any
from .base import CalibrationPattern


class CalibrationPatternManager:
    """Manager for automatic discovery and registration of calibration patterns."""
    
    def __init__(self):
        """Initialize calibration pattern manager."""
        self.patterns: Dict[str, Type[CalibrationPattern]] = {}
        self._discover_patterns()
    
    def _discover_patterns(self):
        """Automatically discover and register all pattern classes."""
        patterns_dir = os.path.dirname(__file__)
        
        # Scan all Python files in the calibration_patterns directory
        for filename in os.listdir(patterns_dir):
            if (filename.endswith('.py') and 
                not filename.startswith('_') and 
                filename not in ['base.py', 'manager.py']):
                
                module_name = filename[:-3]  # Remove .py extension
                
                try:
                    # Import the module
                    module = importlib.import_module(f'.{module_name}', __package__)
                    
                    # Find pattern classes in the module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, CalibrationPattern) and 
                            obj != CalibrationPattern and
                            hasattr(obj, 'PATTERN_INFO')):
                            
                            pattern_id = obj.PATTERN_INFO['id']
                            self.patterns[pattern_id] = obj
                            print(f"🔍 Discovered pattern: {pattern_id} ({obj.__name__})")
                            
                except ImportError as e:
                    print(f"⚠️  Could not import pattern module {module_name}: {e}")
                except Exception as e:
                    print(f"❌ Error processing pattern module {module_name}: {e}")
    
    def register_pattern_type(self, pattern_id: str, pattern_class: Type[CalibrationPattern]):
        """
        Manually register a pattern type.
        
        Args:
            pattern_id: Unique identifier for the pattern
            pattern_class: Pattern class to register
        """
        self.patterns[pattern_id] = pattern_class
        print(f"📝 Registered pattern: {pattern_id} ({pattern_class.__name__})")
    
    def get_pattern_class(self, pattern_id: str) -> Type[CalibrationPattern]:
        """
        Get pattern class by ID.
        
        Args:
            pattern_id: Pattern identifier
            
        Returns:
            Pattern class
            
        Raises:
            ValueError: If pattern type is not found
        """
        if pattern_id not in self.patterns:
            raise ValueError(f"Unknown pattern type: {pattern_id}. "
                           f"Available types: {list(self.patterns.keys())}")
        return self.patterns[pattern_id]
    
    def create_pattern(self, pattern_id: str, **kwargs) -> CalibrationPattern:
        """
        Create a pattern instance.
        
        Args:
            pattern_id: Pattern identifier
            **kwargs: Pattern-specific parameters
            
        Returns:
            CalibrationPattern instance
        """
        pattern_class = self.get_pattern_class(pattern_id)
        return pattern_class(**kwargs)
    
    def get_available_patterns(self) -> Dict[str, Type[CalibrationPattern]]:
        """Get all available pattern types."""
        return self.patterns.copy()
    
    def get_pattern_configurations(self) -> Dict[str, Any]:
        """Get configuration schemas for all discovered patterns."""
        configurations = {}
        
        for pattern_id, pattern_class in self.patterns.items():
            try:
                if hasattr(pattern_class, 'get_configuration_schema'):
                    config = pattern_class.get_configuration_schema()
                    
                    # Add pattern info if available
                    if hasattr(pattern_class, 'PATTERN_INFO'):
                        pattern_info = pattern_class.PATTERN_INFO
                        config.update({
                            'id': pattern_info.get('id', pattern_id),
                            'category': pattern_info.get('category', 'general'),
                            'icon': pattern_info.get('icon', '❓')
                        })
                    
                    configurations[pattern_id] = config
                else:
                    # Fallback configuration
                    configurations[pattern_id] = {
                        'name': pattern_class.__name__,
                        'description': pattern_class.__doc__ or "No description available",
                        'icon': '❓',
                        'category': 'general',
                        'parameters': []
                    }
                    
            except Exception as e:
                print(f"⚠️  Error getting configuration for {pattern_id}: {e}")
        
        return configurations


# Global pattern manager instance
_pattern_manager = None

def get_pattern_manager() -> CalibrationPatternManager:
    """Get the global pattern manager instance."""
    global _pattern_manager
    if _pattern_manager is None:
        _pattern_manager = CalibrationPatternManager()
    return _pattern_manager
