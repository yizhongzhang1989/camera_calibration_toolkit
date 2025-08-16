/**
 * Simplified Chessboard Configuration Module
 * Focused on core functionality with extensive debugging
 */

class SimpleChessboardConfig {
    constructor() {
        console.log('SimpleChessboardConfig constructor called');
        
        this.config = {
            cornerX: 11,
            cornerY: 8,
            squareSize: 0.02
        };
        
        console.log('SimpleChessboardConfig initialized');
    }
    
    // Show the modal
    showModal() {
        console.log('SimpleChessboardConfig.showModal() called');
        const modal = document.getElementById('chessboardModal');
        if (modal) {
            console.log('Modal found, showing...');
            modal.style.display = 'block';
            
            // Fill in current values
            const modalX = document.getElementById('modal-chessboard-x');
            const modalY = document.getElementById('modal-chessboard-y');
            const modalSize = document.getElementById('modal-square-size');
            
            if (modalX) {
                modalX.value = this.config.cornerX;
                console.log(`Set modalX to ${this.config.cornerX}`);
            } else {
                console.warn('modal-chessboard-x element not found');
            }
            
            if (modalY) {
                modalY.value = this.config.cornerY;
                console.log(`Set modalY to ${this.config.cornerY}`);
            } else {
                console.warn('modal-chessboard-y element not found');
            }
            
            if (modalSize) {
                modalSize.value = this.config.squareSize;
                console.log(`Set modalSize to ${this.config.squareSize}`);
            } else {
                console.warn('modal-square-size element not found');
            }
            
            // Update the display and canvas
            this.updateDisplay();
            this.drawCanvas();
            
        } else {
            console.error('Modal with id "chessboardModal" not found');
        }
    }
    
    // Hide the modal
    hideModal() {
        console.log('SimpleChessboardConfig.hideModal() called');
        const modal = document.getElementById('chessboardModal');
        if (modal) {
            modal.style.display = 'none';
            console.log('Modal hidden');
        } else {
            console.error('Modal not found');
        }
    }
    
    // Update the display elements
    updateDisplay() {
        console.log('SimpleChessboardConfig.updateDisplay() called');
        
        const dimensions = document.getElementById('board-dimensions');
        const squareSize = document.getElementById('board-square-size');
        
        if (dimensions) {
            dimensions.textContent = `${this.config.cornerX} × ${this.config.cornerY} corners`;
            console.log(`Updated dimensions: ${this.config.cornerX} × ${this.config.cornerY} corners`);
        } else {
            console.warn('board-dimensions element not found');
        }
        
        if (squareSize) {
            const sizeInMm = (this.config.squareSize * 1000).toFixed(1);
            squareSize.textContent = `${sizeInMm} mm`;
            console.log(`Updated square size: ${sizeInMm} mm`);
        } else {
            console.warn('board-square-size element not found');
        }
    }
    
    // Draw the chessboard preview
    drawCanvas() {
        console.log('SimpleChessboardConfig.drawCanvas() called');
        
        const canvas = document.getElementById('chessboard-canvas');
        if (!canvas) {
            console.error('Canvas with id "chessboard-canvas" not found');
            return;
        }
        
        console.log('Canvas found, drawing chessboard preview');
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Calculate dimensions
        const squaresX = this.config.cornerX + 1;
        const squaresY = this.config.cornerY + 1;
        
        // Calculate square size to fit in canvas
        const padding = 10;
        const availableWidth = width - 2 * padding;
        const availableHeight = height - 2 * padding;
        
        const squareSize = Math.min(
            availableWidth / squaresX,
            availableHeight / squaresY
        );
        
        const totalWidth = squareSize * squaresX;
        const totalHeight = squareSize * squaresY;
        const startX = (width - totalWidth) / 2;
        const startY = (height - totalHeight) / 2;
        
        console.log(`Drawing ${squaresX}×${squaresY} chessboard, square size: ${squareSize.toFixed(1)}px`);
        
        // Draw chessboard
        for (let row = 0; row < squaresY; row++) {
            for (let col = 0; col < squaresX; col++) {
                const isBlack = (row + col) % 2 === 0;
                ctx.fillStyle = isBlack ? '#000000' : '#FFFFFF';
                
                const x = startX + col * squareSize;
                const y = startY + row * squareSize;
                
                ctx.fillRect(x, y, squareSize, squareSize);
            }
        }
        
        // Draw border
        ctx.strokeStyle = '#666666';
        ctx.lineWidth = 1;
        ctx.strokeRect(startX, startY, totalWidth, totalHeight);
        
        console.log('Chessboard drawing completed');
    }
    
    // Save configuration from modal
    saveConfig() {
        console.log('SimpleChessboardConfig.saveConfig() called');
        
        const modalX = document.getElementById('modal-chessboard-x');
        const modalY = document.getElementById('modal-chessboard-y');
        const modalSize = document.getElementById('modal-square-size');
        
        if (modalX && modalY && modalSize) {
            this.config.cornerX = parseInt(modalX.value) || 11;
            this.config.cornerY = parseInt(modalY.value) || 8;
            this.config.squareSize = parseFloat(modalSize.value) || 0.02;
            
            console.log('Config saved:', this.config);
            
            // Update hidden form fields if they exist
            const hiddenX = document.getElementById('chessboard-x');
            const hiddenY = document.getElementById('chessboard-y');
            const hiddenSize = document.getElementById('square-size');
            
            if (hiddenX) hiddenX.value = this.config.cornerX;
            if (hiddenY) hiddenY.value = this.config.cornerY;
            if (hiddenSize) hiddenSize.value = this.config.squareSize;
            
            // Update display
            this.updateDisplay();
            this.drawCanvas();
            
            // Hide modal
            this.hideModal();
            
            console.log('Configuration saved and applied successfully');
            
        } else {
            console.error('Could not find modal input elements');
        }
    }
}
