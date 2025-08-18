/**
 * Chessboard Configuration Module
 * Handles chessboard pattern configuration and display using Python-generated images
 */

class ChessboardConfig {
    constructor(options = {}) {
        console.log('ChessboardConfig constructor called with options:', options);
        
        // Default configuration
        this.config = {
            cornerX: 11,
            cornerY: 8,
            squareSize: 0.020, // meters  
            patternType: 'standard',
            charucoSquareX: 5,
            charucoSquareY: 7,
            charucoSquareSize: 0.040,
            charucoMarkerSize: 0.020,
            charucoDictionary: 'DICT_6X6_250'
        };
        
        console.log('ChessboardConfig initialized with config:', this.config);
        this.eventListenersInitialized = false;
    }
    
    // Initialize after DOM is ready
    initialize() {
        console.log('ChessboardConfig.initialize() called');
        if (!this.eventListenersInitialized) {
            this.initializeEventListeners();
            this.eventListenersInitialized = true;
        }
        
        // Load from session storage if available
        this.loadFromSession();
        
        // Initial display update
        this.updateDisplay();
    }
    
    initializeEventListeners() {
        console.log('Initializing event listeners...');
        
        // Pattern type change in modal
        document.addEventListener('change', (e) => {
            if (e.target.name === 'pattern-type') {
                this.handlePatternTypeChange(e.target.value);
            }
        });

        // Input changes for all configuration fields
        const inputIds = [
            'modal-chessboard-x', 'modal-chessboard-y', 'modal-square-size',
            'modal-charuco-x', 'modal-charuco-y', 
            'modal-charuco-square-size', 'modal-charuco-marker-size', 'modal-charuco-dictionary'
        ];
        
        inputIds.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('input', () => this.updateModalPreview());
            }
        });

        // Apply button
        document.addEventListener('click', (e) => {
            if (e.target.matches('.btn-apply-config')) {
                this.applyConfiguration();
            }
            
            // Download button
            if (e.target.matches('#download-pattern-btn')) {
                this.downloadPattern();
            }
        });

        // Modal show event
        document.addEventListener('shown.bs.modal', (e) => {
            if (e.target.id === 'chessboard-config-modal') {
                this.loadConfigToModal();
                this.updateModalPreview();
            }
        });
        
        console.log('Event listeners initialization complete');
    }
    
    handlePatternTypeChange(patternType) {
        this.config.patternType = patternType;
        
        // Show/hide appropriate config sections
        const chessboardConfig = document.getElementById('chessboard-config');
        const charucoConfig = document.getElementById('charuco-config');
        
        if (chessboardConfig && charucoConfig) {
            if (patternType === 'chessboard' || patternType === 'standard') {
                chessboardConfig.style.display = 'block';
                charucoConfig.style.display = 'none';
            } else if (patternType === 'charuco') {
                chessboardConfig.style.display = 'none';
                charucoConfig.style.display = 'block';
            }
        }
        
        this.updateModalPreview();
    }
    
    loadFromSession() {
        try {
            const saved = sessionStorage.getItem('chessboard_config');
            if (saved) {
                this.config = { ...this.config, ...JSON.parse(saved) };
                console.log('Loaded config from session:', this.config);
            }
        } catch (error) {
            console.warn('Could not load chessboard config from session:', error);
        }
    }

    saveToSession() {
        try {
            sessionStorage.setItem('chessboard_config', JSON.stringify(this.config));
        } catch (error) {
            console.warn('Could not save chessboard config to session:', error);
        }
    }
    
    loadConfigToModal() {
        // Pattern type radio buttons
        const patternTypeInputs = document.querySelectorAll('input[name="pattern-type"]');
        patternTypeInputs.forEach(input => {
            input.checked = input.value === this.config.patternType;
        });

        // Trigger pattern type change to show correct sections
        this.handlePatternTypeChange(this.config.patternType);

        // Load values into modal inputs
        this.setElementValue('modal-chessboard-x', this.config.cornerX);
        this.setElementValue('modal-chessboard-y', this.config.cornerY);
        this.setElementValue('modal-square-size', this.config.squareSize);
        this.setElementValue('modal-charuco-x', this.config.charucoSquareX);
        this.setElementValue('modal-charuco-y', this.config.charucoSquareY);
        this.setElementValue('modal-charuco-square-size', this.config.charucoSquareSize);
        this.setElementValue('modal-charuco-marker-size', this.config.charucoMarkerSize);
        this.setElementValue('modal-charuco-dictionary', this.config.charucoDictionary);
    }

    setElementValue(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.value = value;
        }
    }
    
    applyConfiguration() {
        // Get current pattern type
        const patternTypeInput = document.querySelector('input[name="pattern-type"]:checked');
        if (patternTypeInput) {
            this.config.patternType = patternTypeInput.value;
        }

        if (this.config.patternType === 'chessboard' || this.config.patternType === 'standard') {
            // Update chessboard config
            this.config.cornerX = this.getCurrentValue('modal-chessboard-x', this.config.cornerX);
            this.config.cornerY = this.getCurrentValue('modal-chessboard-y', this.config.cornerY);
            this.config.squareSize = this.getCurrentValue('modal-square-size', this.config.squareSize);
        } else if (this.config.patternType === 'charuco') {
            // Update ChArUco config
            this.config.charucoSquareX = this.getCurrentValue('modal-charuco-x', this.config.charucoSquareX);
            this.config.charucoSquareY = this.getCurrentValue('modal-charuco-y', this.config.charucoSquareY);
            this.config.charucoSquareSize = this.getCurrentValue('modal-charuco-square-size', this.config.charucoSquareSize);
            this.config.charucoMarkerSize = this.getCurrentValue('modal-charuco-marker-size', this.config.charucoMarkerSize);
            
            const dictElement = document.getElementById('modal-charuco-dictionary');
            if (dictElement) {
                this.config.charucoDictionary = dictElement.value;
            }
        }

        this.saveToSession();
        this.updateDisplay();

        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('chessboard-config-modal'));
        if (modal) {
            modal.hide();
        }
    }
    
    getCurrentValue(elementId, defaultValue) {
        const element = document.getElementById(elementId);
        if (element) {
            const value = element.type === 'number' ? parseFloat(element.value) : parseInt(element.value);
            return isNaN(value) ? defaultValue : value;
        }
        return defaultValue;
    }
    
    updateDisplay() {
        // Update the main chessboard card display
        this.updatePatternImage();
        this.updatePatternDescription();
    }
    
    updatePatternImage() {
        const img = document.getElementById('chessboard-preview-img');
        if (!img) return;
        
        const params = this.getPatternParams();
        const url = `/api/pattern_image?${new URLSearchParams(params)}`;
        
        img.src = url;
        img.onerror = () => {
            console.error('Failed to load pattern image:', url);
            img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIEltYWdlPC90ZXh0Pjwvc3ZnPg==';
        };
    }
    
    async updatePatternDescription() {
        const typeDisplay = document.getElementById('pattern-type-display');
        const specsDisplay = document.getElementById('pattern-specs-display');
        
        if (!typeDisplay || !specsDisplay) return;
        
        try {
            const params = this.getPatternParams();
            const response = await fetch(`/api/pattern_description?${new URLSearchParams(params)}`);
            const data = await response.json();
            
            if (data.error) {
                console.error('Error getting pattern description:', data.error);
                return;
            }
            
            typeDisplay.textContent = data.pattern_name || 'Pattern';
            specsDisplay.innerHTML = data.description || 'No description available';
            
        } catch (error) {
            console.error('Failed to fetch pattern description:', error);
        }
    }
    
    updateModalPreview() {
        // Update preview in modal if it exists
        const modalImg = document.getElementById('chessboard-modal-preview');
        if (!modalImg) return;
        
        const params = this.getModalPatternParams();
        const url = `/api/pattern_image?${new URLSearchParams(params)}`;
        modalImg.src = url;
    }
    
    getPatternParams() {
        const params = {
            width: 300,
            height: 200,
            pixel_per_square: 20,  // Small preview image with 20px per square
            border_pixels: 0       // No border for preview
        };
        
        if (this.config.patternType === 'charuco') {
            params.pattern_type = 'charuco';
            params.corner_x = this.config.charucoSquareX;
            params.corner_y = this.config.charucoSquareY;
            params.square_size = this.config.charucoSquareSize;
            params.marker_size = this.config.charucoMarkerSize;
            params.dictionary_id = this.getDictionaryId(this.config.charucoDictionary);
        } else {
            params.pattern_type = 'standard';
            params.corner_x = this.config.cornerX;
            params.corner_y = this.config.cornerY;
            params.square_size = this.config.squareSize;
        }
        
        return params;
    }
    
    getModalPatternParams() {
        const patternTypeInput = document.querySelector('input[name="pattern-type"]:checked');
        const patternType = patternTypeInput ? patternTypeInput.value : this.config.patternType;
        
        const params = {
            width: 250,
            height: 180,
            pixel_per_square: 20,  // Small modal preview with 20px per square
            border_pixels: 0       // No border for modal preview
        };
        
        if (patternType === 'charuco') {
            params.pattern_type = 'charuco';
            params.corner_x = this.getCurrentValue('modal-charuco-x', this.config.charucoSquareX);
            params.corner_y = this.getCurrentValue('modal-charuco-y', this.config.charucoSquareY);
            params.square_size = this.getCurrentValue('modal-charuco-square-size', this.config.charucoSquareSize);
            params.marker_size = this.getCurrentValue('modal-charuco-marker-size', this.config.charucoMarkerSize);
            
            const dictElement = document.getElementById('modal-charuco-dictionary');
            const dictValue = dictElement ? dictElement.value : this.config.charucoDictionary;
            params.dictionary_id = this.getDictionaryId(dictValue);
        } else {
            params.pattern_type = 'standard';
            params.corner_x = this.getCurrentValue('modal-chessboard-x', this.config.cornerX);
            params.corner_y = this.getCurrentValue('modal-chessboard-y', this.config.cornerY);
            params.square_size = this.getCurrentValue('modal-square-size', this.config.squareSize);
        }
        
        return params;
    }
    
    getDictionaryId(dictionaryName) {
        // Map dictionary names to OpenCV constants
        const dictMap = {
            'DICT_4X4_50': 0,
            'DICT_4X4_100': 1,
            'DICT_4X4_250': 2,
            'DICT_4X4_1000': 3,
            'DICT_5X5_50': 4,
            'DICT_5X5_100': 5,
            'DICT_5X5_250': 6,
            'DICT_5X5_1000': 7,
            'DICT_6X6_50': 8,
            'DICT_6X6_100': 9,
            'DICT_6X6_250': 10,
            'DICT_6X6_1000': 11
        };
        
        return dictMap[dictionaryName] || 10; // Default to DICT_6X6_250
    }
    
    async downloadPattern() {
        try {
            // Get download parameters
            const quality = document.getElementById('download-quality')?.value || 'high';
            const border = document.getElementById('download-border')?.value || 'medium';
            
            // Map quality settings to pixel_per_square values
            const qualityMap = {
                'standard': 100,
                'high': 150,
                'ultra': 200
            };
            
            // Map border settings to border_pixels values
            const borderMap = {
                'none': 0,
                'small': 25,
                'medium': 50,
                'large': 100
            };
            
            // Get pattern parameters from modal
            const params = this.getModalPatternParams();
            
            // Remove the small preview parameters for download
            delete params.pixel_per_square;
            delete params.border_pixels;
            delete params.width;
            delete params.height;
            
            // Set download-specific parameters
            params.pixel_per_square = qualityMap[quality];
            params.border_pixels = borderMap[border];
            
            // Generate download URL
            const url = `/api/pattern_image?${new URLSearchParams(params)}`;
            
            // Create download link with descriptive filename
            const a = document.createElement('a');
            a.href = url;
            a.download = `pattern_${params.pattern_type}_${params.corner_x}x${params.corner_y}_${quality}quality.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            
            // Show success message
            this.showMessage('Pattern downloaded successfully!', 'success');
            
        } catch (error) {
            console.error('Download failed:', error);
            this.showMessage('Download failed: ' + error.message, 'error');
        }
    }
    
    showMessage(message, type = 'info') {
        // Create a simple toast message
        const toast = document.createElement('div');
        toast.className = `alert alert-${type === 'error' ? 'danger' : 'success'} position-fixed`;
        toast.style.top = '20px';
        toast.style.right = '20px';
        toast.style.zIndex = '9999';
        toast.textContent = message;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 3000);
    }
    
    // Public API methods
    getConfig() {
        return { ...this.config };
    }

    setConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        this.saveToSession();
        this.updateDisplay();
    }
    
    // Compatibility methods for legacy code
    updateChessboardDisplay() {
        this.updateDisplay();
    }
    
    openModal() {
        const modal = new bootstrap.Modal(document.getElementById('chessboard-config-modal'));
        modal.show();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM ready, initializing ChessboardConfig');
    window.chessboardConfig = new ChessboardConfig();
    window.chessboardConfig.initialize();
});
