/**
 * Intrinsic Camera Calibration Web Interface (Refactored)
 * Uses shared ChessboardConfig module for chessboard functionality
 */

class IntrinsicCalibration {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.uploadedImages = [];
        this.calibrationResults = null;
        this.currentView = 'images';
        
        this.initializeEventListeners();
        this.initializeModal();
        this.initializeChessboard();
        this.updateUI();
    }
    
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
        
        // Override saveConfiguration to integrate with intrinsic calibration
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
    
    generateSessionId() {
        return 'intrinsic_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeEventListeners() {
        // File upload
        const imageFiles = document.getElementById('image-files');
        const uploadArea = document.getElementById('image-upload-area');
        
        if (imageFiles) {
            imageFiles.addEventListener('change', (e) => this.handleImageUpload(e.target.files));
        }
        
        if (uploadArea) {
            // Drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                this.handleImageUpload(e.dataTransfer.files);
            });
        }
        
        // Parameters change
        ['chessboard-x', 'chessboard-y', 'square-size', 'distortion-model'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', () => this.updateParameters());
            }
        });
        
        // Buttons
        const calibrateBtn = document.getElementById('calibrate-btn');
        const downloadBtn = document.getElementById('download-btn');
        const clearBtn = document.getElementById('clear-all-images-btn');
        
        if (calibrateBtn) calibrateBtn.addEventListener('click', () => this.startCalibration());
        if (downloadBtn) downloadBtn.addEventListener('click', () => this.downloadResults());
        if (clearBtn) clearBtn.addEventListener('click', () => this.clearAllImages());
        
        // View controls
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchView(e.target.dataset.view));
        });
    }
    
    async handleImageUpload(files) {
        const formData = new FormData();
        formData.append('session_id', this.sessionId);
        formData.append('calibration_type', 'intrinsic');
        
        for (let file of files) {
            if (file.type.startsWith('image/')) {
                formData.append('files', file);
            }
        }
        
        try {
            this.showStatus('Uploading images...', 'info');
            
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
            
        } catch (error) {
            this.showStatus(`Upload failed: ${error.message}`, 'error');
        }
    }
    
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
        
        if (!tableBody) return;
        
        // Clear and populate table
        tableBody.innerHTML = '';
        
        // Add select all row
        const selectAllRow = document.createElement('tr');
        selectAllRow.innerHTML = `
            <td class="image-selection-cell">
                <div class="select-all-container">
                    <input type="checkbox" id="select-all-checkbox" class="select-all-checkbox" checked>
                    <label for="select-all-checkbox" class="select-all-label">Select All</label>
                </div>
            </td>
            <td colspan="3" style="text-align: center; color: #666; font-style: italic; padding: 1rem;">
                Use checkboxes to select images for calibration, or × button to remove images
            </td>
        `;
        tableBody.appendChild(selectAllRow);
        
        this.uploadedImages.forEach((file, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
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
                        <button class="image-delete-btn" onclick="intrinsicCalib.removeImage(${index})" title="Remove image">&times;</button>
                    </div>
                </td>
                <td>
                    <div class="comparison-image-container">
                        <img src="${file.url}" alt="Original ${index + 1}" class="comparison-image" loading="lazy" onclick="intrinsicCalib.openModal('${file.url}', 'Original Image ${index + 1}')">
                    </div>
                </td>
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
            tableBody.appendChild(row);
            
            // Load image dimensions for this row
            this.loadImageDimensions(file.url, index);
        });
        
        // Set up select all functionality
        this.setupSelectAllFunctionality();
    }
    
    loadImageDimensions(imageUrl, index) {
        const img = new Image();
        img.onload = () => {
            const resolutionElement = document.getElementById(`resolution-${index}`);
            if (resolutionElement) {
                resolutionElement.textContent = `${img.width} × ${img.height} px`;
            }
        };
        img.onerror = () => {
            const resolutionElement = document.getElementById(`resolution-${index}`);
            if (resolutionElement) {
                resolutionElement.textContent = 'Unable to load';
            }
        };
        img.src = imageUrl;
    }
    
    setupSelectAllFunctionality() {
        const selectAllCheckbox = document.getElementById('select-all-checkbox');
        const imageCheckboxes = document.querySelectorAll('.image-checkbox');
        
        if (selectAllCheckbox) {
            selectAllCheckbox.addEventListener('change', (e) => {
                imageCheckboxes.forEach(checkbox => {
                    checkbox.checked = e.target.checked;
                });
            });
        }
        
        // Update select all when individual checkboxes change
        imageCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                const allChecked = Array.from(imageCheckboxes).every(cb => cb.checked);
                const noneChecked = Array.from(imageCheckboxes).every(cb => !cb.checked);
                
                if (selectAllCheckbox) {
                    selectAllCheckbox.checked = allChecked;
                    selectAllCheckbox.indeterminate = !allChecked && !noneChecked;
                }
            });
        });
    }
    
    async removeImage(index) {
        try {
            const response = await fetch('/api/remove_image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    image_index: index
                })
            });
            
            const result = await response.json();
            
            if (result.error) {
                this.showStatus(`Error: ${result.error}`, 'error');
                return;
            }
            
            this.uploadedImages = result.files;
            this.showStatus('Image removed', 'success');
            this.updateUI();
            this.displayUploadedImages();
            
        } catch (error) {
            this.showStatus(`Remove failed: ${error.message}`, 'error');
        }
    }
    
    async clearAllImages() {
        if (!confirm('Are you sure you want to clear all images?')) {
            return;
        }
        
        try {
            const response = await fetch('/api/clear_images', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });
            
            const result = await response.json();
            
            if (result.error) {
                this.showStatus(`Error: ${result.error}`, 'error');
                return;
            }
            
            this.uploadedImages = [];
            this.calibrationResults = null;
            this.showStatus('All images cleared', 'success');
            this.updateUI();
            this.displayUploadedImages();
            
        } catch (error) {
            this.showStatus(`Clear failed: ${error.message}`, 'error');
        }
    }
    
    async updateParameters() {
        const chessboardX = document.getElementById('chessboard-x');
        const chessboardY = document.getElementById('chessboard-y');
        const squareSize = document.getElementById('square-size');
        const distortionModel = document.getElementById('distortion-model');
        
        if (!chessboardX || !chessboardY || !squareSize || !distortionModel) {
            throw new Error('Missing parameter input elements');
        }
        
        const params = {
            session_id: this.sessionId,
            chessboard_x: parseInt(chessboardX.value),
            chessboard_y: parseInt(chessboardY.value),
            square_size: parseFloat(squareSize.value),
            distortion_model: distortionModel.value
        };
        
        try {
            const response = await fetch('/api/set_parameters', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params)
            });
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(`Parameter error: ${result.error}`);
            }
            
            console.log('Parameters set successfully:', params);
            
        } catch (error) {
            this.showStatus(`Parameter update failed: ${error.message}`, 'error');
        }
    }
    
    async startCalibration() {
        const selectedIndices = this.getSelectedImageIndices();
        
        if (selectedIndices.length < 3) {
            this.showStatus('Please select at least 3 images for calibration', 'error');
            return;
        }
        
        try {
            this.showStatus('Setting parameters and starting calibration...', 'info');
            
            // First, ensure parameters are set
            try {
                await this.updateParameters();
            } catch (paramError) {
                this.showStatus(`Parameter setup failed: ${paramError.message}`, 'error');
                return;
            }
            
            this.showStatus('Starting calibration...', 'info');
            
            const response = await fetch('/api/calibrate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    selected_indices: selectedIndices,
                    distortion_model: document.getElementById('distortion-model').value
                })
            });
            
            const result = await response.json();
            
            if (result.error) {
                this.showStatus(`Calibration error: ${result.error}`, 'error');
                return;
            }
            
            this.calibrationResults = result;
            this.showStatus(result.message, 'success');
            this.updateUI();
            this.displayResults();
            
        } catch (error) {
            this.showStatus(`Calibration failed: ${error.message}`, 'error');
        }
    }
    
    displayResults() {
        if (!this.calibrationResults) return;
        
        // Show calibration metrics panel
        const metricsPanel = document.getElementById('calibration-metrics');
        if (metricsPanel) {
            metricsPanel.style.display = 'block';
        }
        
        // Camera matrix
        const cameraMatrix = this.calibrationResults.camera_matrix;
        document.getElementById('camera-matrix-display').innerHTML = `
            <pre>${this.formatMatrix(cameraMatrix)}</pre>
        `;
        
        // Distortion coefficients - show only relevant ones based on model
        const distCoeffs = this.calibrationResults.distortion_coefficients[0] || this.calibrationResults.distortion_coefficients;
        const distortionModel = document.getElementById('distortion-model').value;
        
        let relevantCoeffs;
        let labels;
        
        switch(distortionModel) {
            case 'standard':
                relevantCoeffs = distCoeffs.slice(0, 5);
                labels = ['k₁', 'k₂', 'p₁', 'p₂', 'k₃'];
                break;
            case 'rational':
                relevantCoeffs = distCoeffs.slice(0, 8);
                labels = ['k₁', 'k₂', 'p₁', 'p₂', 'k₃', 'k₄', 'k₅', 'k₆'];
                break;
            case 'thin_prism':
                relevantCoeffs = distCoeffs.slice(0, 12);
                labels = ['k₁', 'k₂', 'p₁', 'p₂', 'k₃', 'k₄', 'k₅', 'k₆', 's₁', 's₂', 's₃', 's₄'];
                break;
            case 'tilted':
                relevantCoeffs = distCoeffs;
                labels = ['k₁', 'k₂', 'p₁', 'p₂', 'k₃', 'k₄', 'k₅', 'k₆', 's₁', 's₂', 's₃', 's₄', 'τₓ', 'τᵧ'];
                break;
            default:
                relevantCoeffs = distCoeffs;
                labels = distCoeffs.map((_, i) => `coeff${i+1}`);
        }
        
        const coeffDisplay = relevantCoeffs.map((coeff, i) => 
            `${labels[i]}: ${coeff.toFixed(6)}`
        ).join('\n');
        
        document.getElementById('distortion-display').innerHTML = `
            <pre>${coeffDisplay}</pre>
        `;
        
        // Reprojection error - check both possible field names
        const errorDisplay = document.getElementById('error-display');
        const rmsError = this.calibrationResults.rms_error || this.calibrationResults.reprojection_error;
        if (rmsError !== undefined) {
            errorDisplay.innerHTML = `<strong>${rmsError.toFixed(4)} pixels</strong>`;
        } else {
            errorDisplay.innerHTML = '<em>Not calculated</em>';
        }
        
        // Images count
        const imagesCountDisplay = document.getElementById('images-count-display');
        if (imagesCountDisplay && this.calibrationResults.images_used !== undefined) {
            imagesCountDisplay.innerHTML = `<strong>${this.calibrationResults.images_used} images</strong>`;
        }
        
        // Update progress info
        const progressInfo = document.getElementById('progress-info');
        if (progressInfo) {
            progressInfo.innerHTML = `✅ Calibration complete with ${this.calibrationResults.images_used || this.uploadedImages.length} images`;
        }
        
        // Show download button
        const downloadBtn = document.getElementById('download-btn');
        if (downloadBtn) {
            downloadBtn.style.display = 'block';
        }
        
        // Update table with corner detection and undistorted images
        this.updateImageTable();
    }
    
    updateImageTable() {
        if (!this.calibrationResults || !this.calibrationResults.corner_images) return;
        
        const cornerImages = this.calibrationResults.corner_images || [];
        const undistortedImages = this.calibrationResults.undistorted_images || [];
        
        // Create lookup maps by index
        const cornerImageMap = {};
        const undistortedImageMap = {};
        
        cornerImages.forEach(img => {
            cornerImageMap[img.index] = img;
        });
        
        undistortedImages.forEach(img => {
            undistortedImageMap[img.index] = img;
        });
        
        this.uploadedImages.forEach((file, index) => {
            // Update corner detection column
            const cornerCell = document.getElementById(`corner-cell-${index}`);
            if (cornerImageMap[index]) {
                cornerCell.innerHTML = `
                    <div class="comparison-image-container">
                        <img src="${cornerImageMap[index].url}" alt="Corners ${index + 1}" class="comparison-image" loading="lazy" onclick="intrinsicCalib.openModal('${cornerImageMap[index].url}', 'Corner Detection ${index + 1}')">
                    </div>
                `;
            } else {
                cornerCell.innerHTML = `
                    <div style="color: #dc3545; padding: 2rem;">
                        <div style="font-size: 2rem;">❌</div>
                        <div>No corners detected</div>
                    </div>
                `;
            }
            
            // Update undistorted column
            const undistortedCell = document.getElementById(`undistorted-cell-${index}`);
            if (undistortedImageMap[index]) {
                undistortedCell.innerHTML = `
                    <div class="comparison-image-container">
                        <img src="${undistortedImageMap[index].url}" alt="Undistorted ${index + 1}" class="comparison-image" loading="lazy" onclick="intrinsicCalib.openModal('${undistortedImageMap[index].url}', 'Undistorted Image ${index + 1}')">
                    </div>
                `;
            } else {
                undistortedCell.innerHTML = `
                    <div style="color: #666; padding: 2rem;">
                        <div style="font-size: 2rem;">⚠️</div>
                        <div>Undistortion failed</div>
                    </div>
                `;
            }
        });
    }
    
    formatMatrix(matrix) {
        if (!matrix || !Array.isArray(matrix)) return 'Matrix not available';
        
        // Ensure we have a proper 3x3 matrix format
        if (matrix.length === 9) {
            // Convert flat array to 3x3 matrix
            const mat3x3 = [
                matrix.slice(0, 3),
                matrix.slice(3, 6),
                matrix.slice(6, 9)
            ];
            return this.format3x3Matrix(mat3x3);
        } else if (matrix.length === 3 && Array.isArray(matrix[0])) {
            // Already in 3x3 format
            return this.format3x3Matrix(matrix);
        } else {
            return 'Invalid matrix format';
        }
    }
    
    format3x3Matrix(matrix) {
        return matrix.map(row => 
            '[' + row.map(val => val.toFixed(2).padStart(10)).join('  ') + ']'
        ).join('\n');
    }
    
    async downloadResults() {
        if (!this.calibrationResults) {
            this.showStatus('No calibration results to download', 'error');
            return;
        }
        
        try {
            const response = await fetch(`/api/export_results/${this.sessionId}`);
            
            if (!response.ok) {
                throw new Error('Download failed');
            }
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `intrinsic_calibration_results_${this.sessionId}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            this.showStatus('Results downloaded successfully', 'success');
            
        } catch (error) {
            this.showStatus(`Download failed: ${error.message}`, 'error');
        }
    }
    
    updateUI() {
        const imageCount = this.uploadedImages.length;
        const hasResults = this.calibrationResults !== null;
        
        // Update button states
        const calibrateBtn = document.getElementById('calibrate-btn');
        const downloadBtn = document.getElementById('download-btn');
        const clearBtn = document.getElementById('clear-all-images-btn');
        
        if (calibrateBtn) calibrateBtn.disabled = imageCount < 3;
        if (downloadBtn) {
            downloadBtn.disabled = !hasResults;
            downloadBtn.style.display = hasResults ? 'block' : 'none';
        }
        if (clearBtn) clearBtn.style.display = imageCount > 0 ? 'block' : 'none';
        
        // Show/hide calibration metrics panel
        const metricsPanel = document.getElementById('calibration-metrics');
        if (metricsPanel) {
            metricsPanel.style.display = hasResults ? 'block' : 'none';
        }
        
        // Update counters and status
        const statusElements = document.querySelectorAll('.image-count');
        statusElements.forEach(el => el.textContent = imageCount);
    }
    
    switchView(view) {
        this.currentView = view;
        
        // Update button states
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === view);
        });
        
        // Show/hide content based on view
        // Add view switching logic here if needed
    }
    
    openModal(imageSrc, title) {
        const modal = document.getElementById('imageModal');
        const modalImage = document.getElementById('modalImage');
        const modalTitle = document.getElementById('modalTitle');
        
        if (modal && modalImage && modalTitle) {
            modalImage.src = imageSrc;
            modalTitle.textContent = title;
            modal.style.display = 'block';
        }
    }
    
    initializeModal() {
        const modal = document.getElementById('imageModal');
        const modalClose = document.getElementById('modalClose');
        
        if (modalClose) {
            // Close modal when clicking the close button
            modalClose.addEventListener('click', () => {
                if (modal) modal.style.display = 'none';
            });
        }
        
        if (modal) {
            // Close modal when clicking outside the image
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.style.display = 'none';
                }
            });
        }
        
        // Close modal with Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal && modal.style.display === 'block') {
                modal.style.display = 'none';
            }
        });
    }
    
    showStatus(message, type = 'info') {
        const statusDiv = document.getElementById('calibration-status');
        if (statusDiv) {
            statusDiv.className = `status-display ${type}`;
            statusDiv.innerHTML = `<p>${message}</p>`;
        }
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
    
    getSelectedImageIndices() {
        const selectedIndices = [];
        const checkboxes = document.querySelectorAll('.image-checkbox');
        
        checkboxes.forEach(checkbox => {
            if (checkbox.checked) {
                selectedIndices.push(parseInt(checkbox.dataset.index));
            }
        });
        
        return selectedIndices;
    }
}

// Initialize the application
let intrinsicCalib;
document.addEventListener('DOMContentLoaded', () => {
    intrinsicCalib = new IntrinsicCalibration();
});
