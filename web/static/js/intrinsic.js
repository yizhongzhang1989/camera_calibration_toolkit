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
        ['chessboard-x', 'chessboard-y', 'square-size'].forEach(id => {
            document.getElementById(id).addEventListener('change', () => this.updateParameters());
        });
        
        // Buttons
        document.getElementById('calibrate-btn').addEventListener('click', () => this.startCalibration());
        document.getElementById('clear-btn').addEventListener('click', () => this.clearSession());
        document.getElementById('download-btn').addEventListener('click', () => this.downloadResults());
        
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
            
            this.uploadedImages = result.files;
            this.showStatus(`Successfully uploaded ${result.files.length} images`, 'success');
            this.updateUI();
            this.displayUploadedImages();
            
        } catch (error) {
            this.showStatus(`Upload failed: ${error.message}`, 'error');
        }
    }
    
    displayUploadedImages() {
        const imageList = document.getElementById('image-list');
        const placeholder = document.getElementById('results-placeholder');
        const comparisonTable = document.getElementById('image-comparison-table');
        const tableBody = document.getElementById('image-comparison-body');
        
        // Update file list in control panel
        imageList.innerHTML = '';
        this.uploadedImages.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <span class="file-name">📷 ${file.name}</span>
                <button class="remove-btn" onclick="intrinsicCalib.removeImage(${index})">&times;</button>
            `;
            imageList.appendChild(fileItem);
        });
        
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
        this.uploadedImages.forEach((file, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>
                    <div class="comparison-image-container">
                        <img src="${file.url}" alt="Original ${index + 1}" class="comparison-image" loading="lazy" onclick="intrinsicCalib.openModal('${file.url}', 'Original Image ${index + 1}')">
                    </div>
                </td>
                <td id="corner-cell-${index}">
                    <div style="color: #999; padding: 2rem;">
                        <div style="font-size: 2rem;">⏳</div>
                        <div>Corner detection pending</div>
                    </div>
                </td>
                <td id="undistorted-cell-${index}">
                    <div style="color: #999; padding: 2rem;">
                        <div style="font-size: 2rem;">⏳</div>
                        <div>Calibration needed</div>
                    </div>
                </td>
            `;
            tableBody.appendChild(row);
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
            square_size: parseFloat(document.getElementById('square-size').value)
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
        if (this.uploadedImages.length < 3) {
            this.showStatus('Need at least 3 images for calibration', 'error');
            return;
        }
        
        try {
            this.showStatus('Starting intrinsic calibration...', 'info');
            document.getElementById('calibrate-btn').disabled = true;
            
            // Update parameters first
            await this.updateParameters();
            
            const response = await fetch('/api/calibrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    calibration_type: 'intrinsic'
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
        
        // Distortion coefficients
        const distCoeffs = this.calibrationResults.distortion_coefficients[0];
        document.getElementById('distortion-display').innerHTML = `
            <pre>[${distCoeffs.map(x => x.toFixed(6)).join(',\n ')}]</pre>
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
        
        this.uploadedImages.forEach((file, index) => {
            // Update corner detection column
            const cornerCell = document.getElementById(`corner-cell-${index}`);
            if (cornerImages[index]) {
                cornerCell.innerHTML = `
                    <div class="comparison-image-container">
                        <img src="${cornerImages[index].url}" alt="Corners ${index + 1}" class="comparison-image" loading="lazy" onclick="intrinsicCalib.openModal('${cornerImages[index].url}', 'Corner Detection ${index + 1}')">
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
            if (undistortedImages[index]) {
                undistortedCell.innerHTML = `
                    <div class="comparison-image-container">
                        <img src="${undistortedImages[index].url}" alt="Undistorted ${index + 1}" class="comparison-image" loading="lazy" onclick="intrinsicCalib.openModal('${undistortedImages[index].url}', 'Undistorted Image ${index + 1}')">
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
        return matrix.map(row => 
            '[' + row.map(val => val.toFixed(2).padStart(8)).join(' ') + ']'
        ).join('\\n');
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
    
    async clearSession() {
        try {
            await fetch(`/api/clear_session/${this.sessionId}`, { method: 'POST' });
            
            // Reset UI
            this.uploadedImages = [];
            this.calibrationResults = null;
            this.sessionId = this.generateSessionId();
            
            document.getElementById('image-list').innerHTML = '';
            document.getElementById('calibration-metrics').style.display = 'none';
            
            // Reset table view
            const placeholder = document.getElementById('results-placeholder');
            const comparisonTable = document.getElementById('image-comparison-table');
            placeholder.style.display = 'block';
            comparisonTable.style.display = 'none';
            
            // Reset progress info
            const progressInfo = document.getElementById('progress-info');
            progressInfo.innerHTML = 'Upload images to see calibration results';
            
            this.updateUI();
            this.showStatus('Session cleared', 'info');
            
        } catch (error) {
            this.showStatus(`Clear session failed: ${error.message}`, 'error');
        }
    }
    
    updateUI() {
        const hasImages = this.uploadedImages.length > 0;
        const hasResults = this.calibrationResults !== null;
        
        document.getElementById('calibrate-btn').disabled = !hasImages;
        document.getElementById('download-btn').style.display = hasResults ? 'block' : 'none';
        
        // Update calibrate button text
        const calibrateBtn = document.getElementById('calibrate-btn');
        if (hasResults) {
            calibrateBtn.textContent = '🔄 Recalibrate';
            calibrateBtn.disabled = false;
        } else {
            calibrateBtn.textContent = 'Start Calibration';
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
}

// Initialize the application
let intrinsicCalib;
document.addEventListener('DOMContentLoaded', () => {
    intrinsicCalib = new IntrinsicCalibration();
});
