/**
 * Intrinsic Camera Calibration Web Interface
 * Handles file uploads, parameter settings, and calibration workflow
 */

class IntrinsicCalibration {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.uploadedImages = [];
        this.calibrationResults = null;
        this.currentView = 'images';
        
        this.initializeEventListeners();
        this.initializeModal();
        this.updateUI();
    }
    
    generateSessionId() {
        return 'intrinsic_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeEventListeners() {
        // File upload
        const imageFiles = document.getElementById('image-files');
        const uploadArea = document.getElementById('image-upload-area');
        
        imageFiles.addEventListener('change', (e) => this.handleImageUpload(e.target.files));
        
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
        
        // Parameters change
        ['chessboard-x', 'chessboard-y', 'square-size', 'distortion-model'].forEach(id => {
            document.getElementById(id).addEventListener('change', () => this.updateParameters());
        });
        
        // Buttons
        document.getElementById('calibrate-btn').addEventListener('click', () => this.startCalibration());
        document.getElementById('download-btn').addEventListener('click', () => this.downloadResults());
        document.getElementById('clear-all-images-btn').addEventListener('click', () => this.clearAllImages());
        
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
            placeholder.style.display = 'block';
            comparisonTable.style.display = 'none';
            return;
        }
        
        placeholder.style.display = 'none';
        comparisonTable.style.display = 'table';
        
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
    
    removeImage(index) {
        this.uploadedImages.splice(index, 1);
        this.updateUI();
        this.displayUploadedImages();
    }
    
    async updateParameters() {
        const parameters = {
            chessboard_x: parseInt(document.getElementById('chessboard-x').value),
            chessboard_y: parseInt(document.getElementById('chessboard-y').value),
            square_size: parseFloat(document.getElementById('square-size').value),
            distortion_model: document.getElementById('distortion-model').value
        };
        
        try {
            const response = await fetch('/api/set_parameters', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    parameters: parameters
                })
            });
            
            const result = await response.json();
            if (result.error) {
                this.showStatus(`Parameter error: ${result.error}`, 'error');
            }
            
        } catch (error) {
            this.showStatus(`Parameter update failed: ${error.message}`, 'error');
        }
    }
    
    async startCalibration() {
        const selectedIndices = this.getSelectedImageIndices();
        
        if (selectedIndices.length < 3) {
            this.showStatus('Need at least 3 selected images for calibration', 'error');
            return;
        }
        
        try {
            this.showStatus(`Starting intrinsic calibration with ${selectedIndices.length} selected images...`, 'info');
            document.getElementById('calibrate-btn').disabled = true;
            
            // Update parameters first
            await this.updateParameters();
            
            const response = await fetch('/api/calibrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    calibration_type: 'intrinsic',
                    selected_indices: selectedIndices
                })
            });
            
            const result = await response.json();
            
            if (result.error) {
                this.showStatus(`Calibration failed: ${result.error}`, 'error');
                document.getElementById('calibrate-btn').disabled = false;
                return;
            }
            
            this.calibrationResults = result;
            this.showStatus(result.message, 'success');
            this.displayResults();
            this.updateUI();
            
        } catch (error) {
            this.showStatus(`Calibration failed: ${error.message}`, 'error');
            document.getElementById('calibrate-btn').disabled = false;
        }
    }
    
    displayResults() {
        if (!this.calibrationResults) return;
        
        // Show metrics in control panel
        const metricsDiv = document.getElementById('calibration-metrics');
        metricsDiv.style.display = 'block';
        
        // Camera matrix
        const cameraMatrix = this.calibrationResults.camera_matrix;
        document.getElementById('camera-matrix-display').innerHTML = `
            <pre>${this.formatMatrix(cameraMatrix)}</pre>
        `;
        
        // Distortion coefficients - show only relevant ones based on model
        const distCoeffs = this.calibrationResults.distortion_coefficients[0];
        const distortionModel = document.getElementById('distortion-model').value;
        
        let relevantCoeffs;
        let labels;
        
        switch(distortionModel) {
            case 'standard':
                relevantCoeffs = distCoeffs.slice(0, 5);
                labels = ['k1', 'k2', 'p1', 'p2', 'k3'];
                break;
            case 'rational':
                relevantCoeffs = distCoeffs.slice(0, 8);
                labels = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'];
                break;
            case 'thin_prism':
                relevantCoeffs = distCoeffs.slice(0, 12);
                labels = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 's1', 's2', 's3', 's4'];
                break;
            case 'tilted':
                relevantCoeffs = distCoeffs;
                labels = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 's1', 's2', 's3', 's4', 'τx', 'τy'];
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
        
        // Reprojection error
        const errorDisplay = document.getElementById('error-display');
        if (this.calibrationResults.reprojection_error !== undefined) {
            errorDisplay.innerHTML = `<strong>${this.calibrationResults.reprojection_error.toFixed(4)} pixels</strong>`;
        } else {
            errorDisplay.innerHTML = '<em>Not calculated</em>';
        }
        
        // Images count
        document.getElementById('images-count-display').innerHTML = 
            `<strong>${this.uploadedImages.length} images</strong>`;
        
        // Update progress info
        const progressInfo = document.getElementById('progress-info');
        progressInfo.innerHTML = `✅ Calibration complete with ${this.uploadedImages.length} images`;
        
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
    
    switchView(view) {
        // Note: With the new table layout, view switching is no longer needed
        // The table always shows all three views side by side
        this.currentView = view;
        console.log(`View switched to: ${view} (table layout shows all views)`);
    }
    
    async downloadResults() {
        if (!this.calibrationResults) return;
        
        try {
            const response = await fetch(`/api/export_results/${this.sessionId}`);
            const blob = await response.blob();
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `intrinsic_calibration_results_${this.sessionId}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
        } catch (error) {
            this.showStatus(`Download failed: ${error.message}`, 'error');
        }
    }
    
    updateUI() {
        const hasImages = this.uploadedImages.length > 0;
        const hasResults = this.calibrationResults !== null;
        
        document.getElementById('calibrate-btn').disabled = !hasImages;
        document.getElementById('download-btn').style.display = hasResults ? 'block' : 'none';
        document.getElementById('clear-all-images-btn').style.display = hasImages ? 'block' : 'none';
        
        // Update calibrate button text
        const calibrateBtn = document.getElementById('calibrate-btn');
        if (hasResults) {
            calibrateBtn.textContent = '🔄 Recalibrate';
            calibrateBtn.disabled = false;
        } else {
            calibrateBtn.textContent = 'Start Calibration';
        }
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
            document.getElementById('calibration-metrics').style.display = 'none';
            
            const placeholder = document.getElementById('results-placeholder');
            const comparisonTable = document.getElementById('image-comparison-table');
            placeholder.style.display = 'block';
            comparisonTable.style.display = 'none';
            
            const progressInfo = document.getElementById('progress-info');
            progressInfo.innerHTML = 'Upload images to see calibration results';
            
            this.updateUI();
            this.showStatus('All images cleared', 'info');
            
        } catch (error) {
            this.showStatus(`Failed to clear images: ${error.message}`, 'error');
        }
    }
    
    showStatus(message, type = 'info') {
        const statusDiv = document.getElementById('calibration-status');
        statusDiv.className = `status-display ${type}`;
        statusDiv.innerHTML = `<p>${message}</p>`;
        
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
    
    initializeModal() {
        const modal = document.getElementById('imageModal');
        const modalClose = document.getElementById('modalClose');
        
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
        modalClose.onclick = closeModal;
        modal.onclick = (e) => {
            if (e.target === modal) closeModal();
        };
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
}

// Initialize the application
let intrinsicCalib;
document.addEventListener('DOMContentLoaded', () => {
    intrinsicCalib = new IntrinsicCalibration();
});
