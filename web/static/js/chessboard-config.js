/**
 * Chessboard Configuration Module
 * Shared functionality for chessboard target configuration and download
 * Used across intrinsic and hand-eye calibration interfaces
 */

class ChessboardConfig {
    constructor(options = {}) {
        console.log('ChessboardConfig constructor called with options:', options);
        this.modalId = options.modalId || 'chessboardModal';
        this.canvasId = options.canvasId || 'chessboard-canvas';
        this.downloadCanvasId = options.downloadCanvasId || 'download-canvas';
        this.statusCallback = options.statusCallback || this.defaultStatusCallback;
        
        // Default configuration
        this.config = {
            cornerX: 11,
            cornerY: 8,
            squareSize: 0.02, // meters
            paperSize: 'a4',
            customWidth: 210,
            customHeight: 297,
            format: 'png',
            resolution: 300
        };
        
        console.log('ChessboardConfig initialized with config:', this.config);
        
        // Don't initialize event listeners in constructor - wait for DOM to be ready
        this.eventListenersInitialized = false;
    }
    
    // Initialize after DOM is ready
    initialize() {
        console.log('ChessboardConfig.initialize() called');
        if (!this.eventListenersInitialized) {
            this.initializeEventListeners();
            this.eventListenersInitialized = true;
        }
        
        // Initial display update
        this.updateChessboardDisplay();
    }
    
    defaultStatusCallback(message, type) {
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
    
    initializeEventListeners() {
        console.log('Initializing event listeners...');
        
        // Download size change
        const downloadSize = document.getElementById('download-size');
        if (downloadSize) {
            console.log('Found download-size element');
            downloadSize.addEventListener('change', (e) => {
                const customGroup = document.getElementById('custom-size-group');
                if (customGroup) {
                    customGroup.style.display = e.target.value === 'custom' ? 'block' : 'none';
                }
                this.updateDownloadPreview();
            });
        } else {
            console.warn('download-size element not found');
        }
        
        // Download parameters change
        ['custom-width', 'custom-height', 'download-format', 'download-resolution'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                console.log(`Found element: ${id}`);
                element.addEventListener('change', () => this.updateDownloadPreview());
            } else {
                console.warn(`Element not found: ${id}`);
            }
        });
        
        // Modal chessboard parameters change
        ['modal-chessboard-x', 'modal-chessboard-y', 'modal-square-size'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                console.log(`Found element: ${id}`);
                element.addEventListener('input', () => {
                    this.updateChessboardDisplay();
                    this.updateDownloadPreview();
                });
            } else {
                console.warn(`Element not found: ${id}`);
            }
        });
        
        console.log('Event listeners initialization complete');
    }
    
    openModal(currentConfig = {}) {
        console.log('Opening chessboard modal with config:', currentConfig);
        // Sync modal values with current settings
        this.config = { ...this.config, ...currentConfig };
        
        const elements = {
            'modal-chessboard-x': this.config.cornerX,
            'modal-chessboard-y': this.config.cornerY,
            'modal-square-size': this.config.squareSize
        };
        
        console.log('Setting modal element values:', elements);
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.value = value;
                console.log(`Set ${id} to ${value}`);
            } else {
                console.warn(`Element with id ${id} not found`);
            }
        });
        
        // Update previews
        this.updateChessboardDisplay();
        this.updateDownloadPreview();
        
        // Show modal
        const modal = document.getElementById(this.modalId);
        if (modal) {
            modal.style.display = 'block';
            console.log('Modal displayed');
        } else {
            console.error(`Modal with id ${this.modalId} not found`);
        }
    }
    
    closeModal() {
        const modal = document.getElementById(this.modalId);
        if (modal) modal.style.display = 'none';
    }
    
    saveConfiguration(callback) {
        // Get values from modal
        const x = document.getElementById('modal-chessboard-x');
        const y = document.getElementById('modal-chessboard-y');
        const size = document.getElementById('modal-square-size');
        
        if (x && y && size) {
            this.config.cornerX = parseInt(x.value);
            this.config.cornerY = parseInt(y.value);
            this.config.squareSize = parseFloat(size.value);
            
            // Call callback with new configuration
            if (callback) callback(this.config);
            
            this.closeModal();
            this.statusCallback('Chessboard configuration updated', 'success');
        }
    }
    
    updateChessboardDisplay() {
        console.log('updateChessboardDisplay() called');
        const x = this.getCurrentCornerX();
        const y = this.getCurrentCornerY();
        console.log(`Chessboard dimensions: ${x} × ${y}`);
        
        // Update display elements
        const dimensions = document.getElementById('board-dimensions');
        const squareSize = document.getElementById('board-square-size');
        
        if (dimensions) {
            dimensions.textContent = `${x} × ${y} corners`;
            console.log('Updated dimensions display');
        } else {
            console.warn('board-dimensions element not found');
        }
        
        if (squareSize) {
            const size = this.getCurrentSquareSize();
            squareSize.textContent = `${(size * 1000).toFixed(1)} mm`;
            console.log(`Updated square size display: ${(size * 1000).toFixed(1)} mm`);
        } else {
            console.warn('board-square-size element not found');
        }
        
        // Draw chessboard preview
        console.log('Drawing chessboard preview...');
        this.drawChessboardPreview(x, y);
    }
    
    drawChessboardPreview(cornerX, cornerY) {
        console.log(`drawChessboardPreview(${cornerX}, ${cornerY}) called`);
        const canvas = document.getElementById(this.canvasId);
        if (!canvas) {
            console.error(`Canvas with id ${this.canvasId} not found`);
            return;
        }
        
        console.log('Canvas found, drawing chessboard...');
        const ctx = canvas.getContext('2d');
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
        
        // Calculate squares (corners + 1 for each dimension)
        const squaresX = cornerX + 1;
        const squaresY = cornerY + 1;
        console.log(`Drawing ${squaresX} × ${squaresY} squares`);
        
        // Calculate square size to fit in canvas with some padding
        const padding = 10;
        const availableWidth = canvasWidth - 2 * padding;
        const availableHeight = canvasHeight - 2 * padding;
        
        const squareSize = Math.min(
            availableWidth / squaresX,
            availableHeight / squaresY
        );
        
        // Center the chessboard
        const boardWidth = squaresX * squareSize;
        const boardHeight = squaresY * squareSize;
        const startX = (canvasWidth - boardWidth) / 2;
        const startY = (canvasHeight - boardHeight) / 2;
        
        // Draw chessboard pattern
        for (let row = 0; row < squaresY; row++) {
            for (let col = 0; col < squaresX; col++) {
                const x = startX + col * squareSize;
                const y = startY + row * squareSize;
                
                // Alternate colors (true chessboard pattern)
                const isBlack = (row + col) % 2 === 0;
                ctx.fillStyle = isBlack ? '#2c2c2c' : '#f8f8f8';
                ctx.fillRect(x, y, squareSize, squareSize);
                
                // Add subtle border to squares
                ctx.strokeStyle = '#ccc';
                ctx.lineWidth = 0.5;
                ctx.strokeRect(x, y, squareSize, squareSize);
            }
        }
        
        // Draw corner indicators (red dots where corners are detected)
        ctx.fillStyle = '#dc3545';
        const cornerRadius = Math.max(2, squareSize * 0.1);
        
        for (let row = 1; row < squaresY; row++) {
            for (let col = 1; col < squaresX; col++) {
                const x = startX + col * squareSize;
                const y = startY + row * squareSize;
                
                ctx.beginPath();
                ctx.arc(x, y, cornerRadius, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
        
        // Add dimension labels
        ctx.fillStyle = '#666';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        
        // Bottom label (X dimension)
        ctx.fillText(`${cornerX} corners`, canvasWidth / 2, canvasHeight - 2);
        
        // Right label (Y dimension) - rotated
        ctx.save();
        ctx.translate(canvasWidth - 5, canvasHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(`${cornerY} corners`, 0, 0);
        ctx.restore();
    }
    
    updateDownloadPreview() {
        // Get current parameters
        const cornerX = this.getCurrentCornerX();
        const cornerY = this.getCurrentCornerY();
        const squareSize = this.getCurrentSquareSize();
        
        // Get download settings
        const sizeType = this.getDownloadSizeType();
        const { width, height } = this.getPaperDimensions(sizeType);
        
        // Update info display
        const elements = {
            'download-squares': `${cornerX + 1}×${cornerY + 1}`,
            'download-dimensions': `${width}×${height} mm`,
            'download-square-size': `${(squareSize * 1000).toFixed(1)} mm`
        };
        
        Object.entries(elements).forEach(([id, text]) => {
            const element = document.getElementById(id);
            if (element) element.textContent = text;
        });
        
        // Draw preview
        this.drawDownloadPreview(cornerX, cornerY, width, height);
    }
    
    drawDownloadPreview(cornerX, cornerY, paperWidth, paperHeight) {
        const canvas = document.getElementById(this.downloadCanvasId);
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
        
        // Calculate squares (corners + 1 for each dimension)
        const squaresX = cornerX + 1;
        const squaresY = cornerY + 1;
        
        // Calculate aspect ratios
        const paperAspect = paperWidth / paperHeight;
        const canvasAspect = canvasWidth / canvasHeight;
        
        // Fit paper aspect ratio to canvas
        let drawWidth, drawHeight, offsetX, offsetY;
        
        if (paperAspect > canvasAspect) {
            // Paper is wider relative to height
            drawWidth = canvasWidth - 20; // padding
            drawHeight = drawWidth / paperAspect;
            offsetX = 10;
            offsetY = (canvasHeight - drawHeight) / 2;
        } else {
            // Paper is taller relative to width
            drawHeight = canvasHeight - 20; // padding
            drawWidth = drawHeight * paperAspect;
            offsetX = (canvasWidth - drawWidth) / 2;
            offsetY = 10;
        }
        
        // Draw paper background
        ctx.fillStyle = 'white';
        ctx.fillRect(offsetX, offsetY, drawWidth, drawHeight);
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        ctx.strokeRect(offsetX, offsetY, drawWidth, drawHeight);
        
        // Calculate chessboard size that fits on paper with margin
        const margin = Math.min(drawWidth, drawHeight) * 0.1; // 10% margin
        const availableWidth = drawWidth - 2 * margin;
        const availableHeight = drawHeight - 2 * margin;
        
        const squareSize = Math.min(availableWidth / squaresX, availableHeight / squaresY);
        const boardWidth = squaresX * squareSize;
        const boardHeight = squaresY * squareSize;
        
        // Center chessboard on paper
        const boardOffsetX = offsetX + (drawWidth - boardWidth) / 2;
        const boardOffsetY = offsetY + (drawHeight - boardHeight) / 2;
        
        // Draw chessboard pattern (clean, no indicators)
        for (let row = 0; row < squaresY; row++) {
            for (let col = 0; col < squaresX; col++) {
                const x = boardOffsetX + col * squareSize;
                const y = boardOffsetY + row * squareSize;
                
                // Alternate colors (true chessboard pattern)
                const isBlack = (row + col) % 2 === 0;
                ctx.fillStyle = isBlack ? '#000000' : '#ffffff';
                ctx.fillRect(x, y, squareSize, squareSize);
            }
        }
        
        // Draw border around chessboard
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(boardOffsetX, boardOffsetY, boardWidth, boardHeight);
    }
    
    downloadChessboard() {
        // Get parameters
        const cornerX = this.getCurrentCornerX();
        const cornerY = this.getCurrentCornerY();
        const squareSize = this.getCurrentSquareSize();
        
        const sizeType = this.getDownloadSizeType();
        const format = this.getDownloadFormat();
        const resolution = this.getDownloadResolution();
        
        const { width, height } = this.getPaperDimensions(sizeType);
        
        // Create high-resolution canvas
        const pixelWidth = Math.round((width / 25.4) * resolution); // mm to pixels
        const pixelHeight = Math.round((height / 25.4) * resolution);
        
        const canvas = document.createElement('canvas');
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
        const ctx = canvas.getContext('2d');
        
        // Calculate squares and sizing
        const squaresX = cornerX + 1;
        const squaresY = cornerY + 1;
        
        // Calculate chessboard size with margin
        const marginPixels = Math.min(pixelWidth, pixelHeight) * 0.1;
        const availableWidth = pixelWidth - 2 * marginPixels;
        const availableHeight = pixelHeight - 2 * marginPixels;
        
        const squarePixels = Math.min(availableWidth / squaresX, availableHeight / squaresY);
        const boardWidth = squaresX * squarePixels;
        const boardHeight = squaresY * squarePixels;
        
        // Center on canvas
        const offsetX = (pixelWidth - boardWidth) / 2;
        const offsetY = (pixelHeight - boardHeight) / 2;
        
        // Fill background white
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, pixelWidth, pixelHeight);
        
        // Draw chessboard
        for (let row = 0; row < squaresY; row++) {
            for (let col = 0; col < squaresX; col++) {
                const x = offsetX + col * squarePixels;
                const y = offsetY + row * squarePixels;
                
                const isBlack = (row + col) % 2 === 0;
                ctx.fillStyle = isBlack ? '#000000' : '#ffffff';
                ctx.fillRect(x, y, squarePixels, squarePixels);
            }
        }
        
        // Generate download
        const filename = `chessboard_${cornerX + 1}x${cornerY + 1}_${(squareSize * 1000).toFixed(1)}mm.${format}`;
        
        if (format === 'png') {
            canvas.toBlob((blob) => {
                this.downloadBlob(blob, filename);
            }, 'image/png');
        } else if (format === 'svg') {
            this.downloadChessboardSVG(cornerX, cornerY, width, height, squareSize, filename);
        } else if (format === 'pdf') {
            this.statusCallback('PDF format coming soon. Downloading as high-resolution PNG instead.', 'info');
            canvas.toBlob((blob) => {
                this.downloadBlob(blob, filename.replace('.pdf', '.png'));
            }, 'image/png');
        }
        
        this.statusCallback('Chessboard pattern generated successfully!', 'success');
    }
    
    downloadChessboardSVG(cornerX, cornerY, paperWidth, paperHeight, squareSize, filename) {
        const squaresX = cornerX + 1;
        const squaresY = cornerY + 1;
        
        // Calculate board size with margin
        const margin = Math.min(paperWidth, paperHeight) * 0.1;
        const availableWidth = paperWidth - 2 * margin;
        const availableHeight = paperHeight - 2 * margin;
        
        const squareSize_mm = Math.min(availableWidth / squaresX, availableHeight / squaresY);
        const boardWidth = squaresX * squareSize_mm;
        const boardHeight = squaresY * squareSize_mm;
        
        const offsetX = (paperWidth - boardWidth) / 2;
        const offsetY = (paperHeight - boardHeight) / 2;
        
        let svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${paperWidth}mm" height="${paperHeight}mm" viewBox="0 0 ${paperWidth} ${paperHeight}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="white"/>`;
        
        // Generate squares
        for (let row = 0; row < squaresY; row++) {
            for (let col = 0; col < squaresX; col++) {
                const x = offsetX + col * squareSize_mm;
                const y = offsetY + row * squareSize_mm;
                
                const isBlack = (row + col) % 2 === 0;
                if (isBlack) {
                    svg += `\n  <rect x="${x}" y="${y}" width="${squareSize_mm}" height="${squareSize_mm}" fill="black"/>`;
                }
            }
        }
        
        svg += '\n</svg>';
        
        const blob = new Blob([svg], { type: 'image/svg+xml' });
        this.downloadBlob(blob, filename);
    }
    
    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }
    
    // Helper methods to get current values
    getCurrentCornerX() {
        const element = document.getElementById('modal-chessboard-x');
        return element ? parseInt(element.value) || this.config.cornerX : this.config.cornerX;
    }
    
    getCurrentCornerY() {
        const element = document.getElementById('modal-chessboard-y');
        return element ? parseInt(element.value) || this.config.cornerY : this.config.cornerY;
    }
    
    getCurrentSquareSize() {
        const element = document.getElementById('modal-square-size');
        return element ? parseFloat(element.value) || this.config.squareSize : this.config.squareSize;
    }
    
    getDownloadSizeType() {
        const element = document.getElementById('download-size');
        return element ? element.value : this.config.paperSize;
    }
    
    getDownloadFormat() {
        const element = document.getElementById('download-format');
        return element ? element.value : this.config.format;
    }
    
    getDownloadResolution() {
        const element = document.getElementById('download-resolution');
        return element ? parseInt(element.value) : this.config.resolution;
    }
    
    getPaperDimensions(sizeType) {
        switch (sizeType) {
            case 'a4':
                return { width: 210, height: 297 };
            case 'letter':
                return { width: 216, height: 279 };
            case 'custom':
                const widthEl = document.getElementById('custom-width');
                const heightEl = document.getElementById('custom-height');
                return {
                    width: widthEl ? parseInt(widthEl.value) || this.config.customWidth : this.config.customWidth,
                    height: heightEl ? parseInt(heightEl.value) || this.config.customHeight : this.config.customHeight
                };
            default:
                return { width: 210, height: 297 };
        }
    }
    
    // Public API methods
    getConfiguration() {
        return {
            cornerX: this.getCurrentCornerX(),
            cornerY: this.getCurrentCornerY(),
            squareSize: this.getCurrentSquareSize()
        };
    }
    
    setConfiguration(config) {
        this.config = { ...this.config, ...config };
        this.updateChessboardDisplay();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChessboardConfig;
} else if (typeof window !== 'undefined') {
    window.ChessboardConfig = ChessboardConfig;
}
