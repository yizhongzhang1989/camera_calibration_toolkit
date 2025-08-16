/**
 * Base Calibration Class
 * ======================
 * 
 * Shared functionality for all calibration types (intrinsic, eye-in-hand, etc.)
 * Provides common methods for image handling, UI management, and API interactions.
 */

class BaseCalibration {
    constructor(calibrationType) {
        this.calibrationType = calibrationType;
        this.sessionId = this.generateSessionId();
        this.uploadedImages = [];
        this.calibrationResults = null;
        this.currentView = 'images';
        
        this.initializeModal();
        this.initializeChessboard();
    }
    
    // ========================================
    // Session and ID Management
    // ========================================
    
    generateSessionId() {
        return `${this.calibrationType}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    // ========================================
    // Chessboard Configuration Integration
    // ========================================
    
    initializeChessboard() {
        console.log('Initializing chessboard using shared ChessboardConfig module...');
        
        // Create instance of shared ChessboardConfig
        this.chessboardConfig = new ChessboardConfig({
            statusCallback: (message, type) => this.showStatus(message, type)
        });
        
        // Initialize the chessboard config
        this.chessboardConfig.initialize();
        
        // Set up global access for template onclick handlers
        window.chessboardConfig = this.chessboardConfig;
        
        // Override saveConfiguration to integrate with calibration
        const originalSaveConfiguration = this.chessboardConfig.saveConfiguration.bind(this.chessboardConfig);
        this.chessboardConfig.saveConfiguration = (callback) => {
            originalSaveConfiguration((config) => {
                // Update hidden form inputs for backward compatibility
                document.getElementById('chessboard-x').value = config.cornerX;
                document.getElementById('chessboard-y').value = config.cornerY;
                document.getElementById('square-size').value = config.squareSize;
                
                // Update parameters on server
                this.updateParameters();
                
                // Call provided callback if any
                if (callback) callback(config);
                
                this.showStatus('Chessboard configuration updated', 'success');
            });
        };
        
        // Initial display update
        this.updateChessboardDisplay();
    }

    updateChessboardDisplay() {
        // Update the chessboard display using the shared module
        if (this.chessboardConfig && this.chessboardConfig.updateChessboardDisplay) {
            this.chessboardConfig.updateChessboardDisplay();
        }
    }
    
    // ========================================
    // Image Upload Handling
    // ========================================
    
    async handleImageUpload(files) {
        const formData = new FormData();
        formData.append('session_id', this.sessionId);
        formData.append('calibration_type', this.calibrationType);
        
        for (let file of files) {
            if (file.type.startsWith('image/')) {
                formData.append('files', file);
            }
        }
        
        try {
            this.showStatus(`Uploading ${this.getImageUploadMessage()}...`, 'info');
            
            const response = await fetch('/api/upload_images', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.error) {
                this.showStatus(`Error: ${result.error}`, 'error');
                return;
            }
            
            // Backend returns all images (existing + new), so just replace the array
            const allImages = result.files || [];
            this.uploadedImages = allImages;
            
            this.showStatus(result.message, 'success');
            this.updateUI();
            this.displayUploadedImages();
            this.onImagesUploaded(); // Hook for subclasses
            
        } catch (error) {
            this.showStatus(`Upload failed: ${error.message}`, 'error');
        }
    }
    
    // Hook methods for subclasses to override
    getImageUploadMessage() {
        return 'images';
    }
    
    onImagesUploaded() {
        // Override in subclasses if needed
    }
    
    // ========================================
    // Image Management
    // ========================================
    
    removeImage(index) {
        this.uploadedImages.splice(index, 1);
        this.updateUI();
        this.displayUploadedImages();
        this.onImagesChanged(); // Hook for subclasses
    }
    
    async clearAllImages() {
        if (this.uploadedImages.length === 0) return;
        
        if (!confirm(`Are you sure you want to remove all ${this.uploadedImages.length} images?`)) {
            return;
        }
        
        try {
            // Clear from backend session
            await fetch(`/api/clear_session/${this.sessionId}`, { method: 'POST' });
            
            // Reset local data
            this.uploadedImages = [];
            this.calibrationResults = null;
            this.sessionId = this.generateSessionId();
            
            // Reset UI
            this.resetUIAfterClear();
            
            this.updateUI();
            this.showStatus('All images cleared', 'info');
            this.onImagesChanged(); // Hook for subclasses
            
        } catch (error) {
            this.showStatus(`Failed to clear images: ${error.message}`, 'error');
        }
    }
    
    onImagesChanged() {
        // Override in subclasses if needed
    }
    
    resetUIAfterClear() {
        // Reset calibration metrics display
        const metricsElement = document.getElementById('calibration-metrics');
        if (metricsElement) {
            metricsElement.style.display = 'none';
        }
        
        // Reset results display
        const placeholder = document.getElementById('results-placeholder');
        const comparisonTable = document.getElementById('image-comparison-table');
        if (placeholder) placeholder.style.display = 'block';
        if (comparisonTable) comparisonTable.style.display = 'none';
        
        // Reset progress info
        const progressInfo = document.getElementById('progress-info');
        if (progressInfo) {
            progressInfo.innerHTML = this.getDefaultProgressMessage();
        }
    }
    
    getDefaultProgressMessage() {
        return 'Upload images to see calibration results';
    }
    
    // ========================================
    // Parameter Management (Abstract)
    // ========================================
    
    async updateParameters() {
        // Abstract method - must be implemented by subclasses
        throw new Error('updateParameters() must be implemented by subclass');
    }
    
    // ========================================
    // Calibration Process (Abstract)
    // ========================================
    
    async startCalibration() {
        // Abstract method - must be implemented by subclasses
        throw new Error('startCalibration() must be implemented by subclass');
    }
    
    // ========================================
    // Results Management
    // ========================================
    
    async downloadResults() {
        if (!this.calibrationResults) return;
        
        try {
            const response = await fetch(`/api/export_results/${this.sessionId}`);
            const blob = await response.blob();
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${this.calibrationType}_calibration_results_${this.sessionId}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
        } catch (error) {
            this.showStatus(`Download failed: ${error.message}`, 'error');
        }
    }
    
    // ========================================
    // Image Display and UI Management
    // ========================================
    
    displayUploadedImages() {
        const placeholder = document.getElementById('results-placeholder');
        const comparisonTable = document.getElementById('image-comparison-table');
        const tableBody = document.getElementById('image-comparison-body');
        
        // Update table display
        if (this.uploadedImages.length === 0) {
            if (placeholder) placeholder.style.display = 'block';
            if (comparisonTable) comparisonTable.style.display = 'none';
            return;
        }
        
        if (placeholder) placeholder.style.display = 'none';
        if (comparisonTable) comparisonTable.style.display = 'table';
        
        // Clear and populate table
        if (tableBody) {
            tableBody.innerHTML = '';
            
            // Add select all row
            this.addSelectAllRow(tableBody);
            
            // Add image rows
            this.uploadedImages.forEach((file, index) => {
                this.addImageRow(tableBody, file, index);
            });
            
            // Set up event listeners
            this.setupImageTableEventListeners();
        }
    }
    
    addSelectAllRow(tableBody) {
        const selectAllRow = document.createElement('tr');
        selectAllRow.innerHTML = `
            <td class="image-selection-cell">
                <div class="select-all-container">
                    <input type="checkbox" id="select-all-checkbox" class="select-all-checkbox" checked>
                    <label for="select-all-checkbox" class="select-all-label">Select All</label>
                </div>
            </td>
            <td colspan="${this.getTableColumnCount() - 1}" style="text-align: center; color: #666; font-style: italic; padding: 1rem;">
                Use checkboxes to select images for calibration, or × button to remove images
            </td>
        `;
        tableBody.appendChild(selectAllRow);
    }
    
    addImageRow(tableBody, file, index) {
        const row = document.createElement('tr');
        row.innerHTML = this.getImageRowHTML(file, index);
        tableBody.appendChild(row);
        
        // Load image dimensions for this row
        this.loadImageDimensions(file.url, index);
    }
    
    getImageRowHTML(file, index) {
        // Base implementation - can be overridden by subclasses
        const columnCount = this.getTableColumnCount();
        const additionalColumns = this.getAdditionalColumns(index);
        
        return `
            <td class="image-selection-cell">
                <div class="selection-row-1">
                    <span class="image-index">${index + 1}</span>
                    <span class="image-name">${file.name}</span>
                </div>
                <div class="selection-row-2">
                    <span class="image-resolution" id="resolution-${index}">Loading...</span>
                </div>
                <div class="selection-row-3">
                    <input type="checkbox" id="image-${index}" class="image-checkbox" data-index="${index}" checked>
                    <button class="image-delete-btn" onclick="${this.getGlobalInstanceName()}.removeImage(${index})" title="Remove image">&times;</button>
                </div>
            </td>
            <td>
                <div class="comparison-image-container">
                    <img src="${file.url}" alt="Original ${index + 1}" class="comparison-image" loading="lazy" onclick="${this.getGlobalInstanceName()}.openModal('${file.url}', 'Original Image ${index + 1}')">
                </div>
            </td>
            ${additionalColumns}
        `;
    }
    
    getTableColumnCount() {
        return 4; // Base: selection, original, corner, undistorted
    }
    
    getAdditionalColumns(index) {
        // Base implementation for corner detection and undistorted columns
        return `
            <td id="corner-cell-${index}">
                <div class="image-placeholder">
                    <div class="placeholder-icon">⏳</div>
                    <div class="placeholder-text">Corner detection pending</div>
                </div>
            </td>
            <td id="undistorted-cell-${index}">
                <div class="image-placeholder">
                    <div class="placeholder-icon">⏳</div>
                    <div class="placeholder-text">Calibration needed</div>
                </div>
            </td>
        `;
    }
    
    getGlobalInstanceName() {
        // Must be implemented by subclasses to match their global variable name
        return 'baseCalib';
    }
    
    setupImageTableEventListeners() {
        // Add select all functionality
        const selectAllCheckbox = document.getElementById('select-all-checkbox');
        if (selectAllCheckbox) {
            selectAllCheckbox.addEventListener('change', (e) => {
                const imageCheckboxes = document.querySelectorAll('.image-checkbox');
                imageCheckboxes.forEach(checkbox => {
                    checkbox.checked = e.target.checked;
                });
            });
        }
        
        // Add individual checkbox listeners
        const imageCheckboxes = document.querySelectorAll('.image-checkbox');
        imageCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateSelectAllState();
            });
        });
    }
    
    loadImageDimensions(imageUrl, index) {
        const img = new Image();
        img.onload = () => {
            const resolutionElement = document.getElementById(`resolution-${index}`);
            if (resolutionElement) {
                resolutionElement.textContent = `${img.width} × ${img.height}`;
            }
        };
        img.onerror = () => {
            const resolutionElement = document.getElementById(`resolution-${index}`);
            if (resolutionElement) {
                resolutionElement.textContent = 'Resolution unknown';
            }
        };
        img.src = imageUrl;
    }
    
    updateSelectAllState() {
        const selectAllCheckbox = document.getElementById('select-all-checkbox');
        const imageCheckboxes = document.querySelectorAll('.image-checkbox');
        
        if (!selectAllCheckbox || imageCheckboxes.length === 0) return;
        
        const checkedCount = Array.from(imageCheckboxes).filter(cb => cb.checked).length;
        
        if (checkedCount === 0) {
            selectAllCheckbox.checked = false;
            selectAllCheckbox.indeterminate = false;
        } else if (checkedCount === imageCheckboxes.length) {
            selectAllCheckbox.checked = true;
            selectAllCheckbox.indeterminate = false;
        } else {
            selectAllCheckbox.checked = false;
            selectAllCheckbox.indeterminate = true;
        }
    }
    
    getSelectedImageIndices() {
        const imageCheckboxes = document.querySelectorAll('.image-checkbox');
        const selectedIndices = [];
        
        imageCheckboxes.forEach(checkbox => {
            if (checkbox.checked) {
                selectedIndices.push(parseInt(checkbox.dataset.index));
            }
        });
        
        return selectedIndices;
    }
    
    // ========================================
    // UI State Management
    // ========================================
    
    updateUI() {
        const hasImages = this.uploadedImages.length > 0;
        const hasResults = this.calibrationResults !== null;
        
        this.updateButtons(hasImages, hasResults);
        this.updateCalibrationButton(hasImages, hasResults);
        this.onUIUpdate(hasImages, hasResults); // Hook for subclasses
    }
    
    updateButtons(hasImages, hasResults) {
        const downloadBtn = document.getElementById('download-btn');
        const clearImagesBtn = document.getElementById('clear-all-images-btn');
        
        if (downloadBtn) downloadBtn.style.display = hasResults ? 'block' : 'none';
        if (clearImagesBtn) clearImagesBtn.style.display = hasImages ? 'block' : 'none';
    }
    
    updateCalibrationButton(hasImages, hasResults) {
        const calibrateBtn = document.getElementById('calibrate-btn');
        if (!calibrateBtn) return;
        
        const canCalibrate = this.canStartCalibration(hasImages);
        
        if (hasResults) {
            calibrateBtn.textContent = '🔄 Recalibrate';
            calibrateBtn.disabled = !canCalibrate;
        } else {
            calibrateBtn.textContent = this.getCalibrationButtonText();
            calibrateBtn.disabled = !canCalibrate;
        }
    }
    
    canStartCalibration(hasImages) {
        return hasImages; // Base implementation - can be overridden
    }
    
    getCalibrationButtonText() {
        return `Start ${this.calibrationType} Calibration`;
    }
    
    onUIUpdate(hasImages, hasResults) {
        // Hook for subclasses
    }
    
    // ========================================
    // Modal Management
    // ========================================
    
    initializeModal() {
        const modal = document.getElementById('imageModal');
        const modalClose = document.getElementById('modalClose');
        
        if (!modal || !modalClose) return;
        
        // Close modal when clicking the close button
        modalClose.addEventListener('click', () => {
            modal.style.display = 'none';
        });
        
        // Close modal when clicking outside the image
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
        
        // Close modal with Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.style.display === 'block') {
                modal.style.display = 'none';
            }
        });
    }
    
    openModal(imageSrc, title) {
        const modal = document.getElementById('imageModal');
        const modalImage = document.getElementById('modalImage');
        const modalTitle = document.getElementById('modalTitle');
        
        if (!modal || !modalImage || !modalTitle) return;
        
        modalImage.src = imageSrc;
        modalTitle.textContent = title;
        modal.style.display = 'block';
        
        // Prevent scrolling on the background
        document.body.style.overflow = 'hidden';
        
        // Restore scrolling when modal is closed
        const closeModal = () => {
            document.body.style.overflow = '';
            modal.style.display = 'none';
        };
        
        // Update close functionality to restore scrolling
        const modalClose = document.getElementById('modalClose');
        if (modalClose) modalClose.onclick = closeModal;
        modal.onclick = (e) => {
            if (e.target === modal) closeModal();
        };
    }
    
    // ========================================
    // Status and Messaging
    // ========================================
    
    showStatus(message, type = 'info') {
        const statusDiv = document.getElementById('calibration-status');
        if (statusDiv) {
            statusDiv.className = `status-display ${type}`;
            statusDiv.innerHTML = `<p>${message}</p>`;
        }
        
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
}
