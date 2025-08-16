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
        const resultsGrid = document.getElementById('results-grid');
        
        // Update file list
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
        
        // Update results grid with original images
        resultsGrid.innerHTML = '';
        this.uploadedImages.forEach((file, index) => {
            const imageDiv = document.createElement('div');
            imageDiv.className = 'result-image';
            imageDiv.innerHTML = `
                <img src="${file.url}" alt="Image ${index}" loading="lazy">
                <div class="image-info">
                    <span class="image-index">${index + 1}</span>
                    <span class="image-name">${file.name}</span>
                </div>
            `;
            resultsGrid.appendChild(imageDiv);
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
        
        // Show metrics
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
            <pre>[${distCoeffs.map(x => x.toFixed(6)).join(',\\n ')}]</pre>
        `;
        
        // Reprojection error (if available)
        const errorDisplay = document.getElementById('error-display');
        if (this.calibrationResults.reprojection_error !== undefined) {
            errorDisplay.innerHTML = `<strong>${this.calibrationResults.reprojection_error.toFixed(4)} pixels</strong>`;
        } else {
            errorDisplay.innerHTML = '<em>Not calculated</em>';
        }
        
        // Images count
        document.getElementById('images-count-display').innerHTML = 
            `<strong>${this.uploadedImages.length} images</strong>`;
            
        // Update results summary
        document.getElementById('results-summary').innerHTML = `
            <div class="success-message">
                <strong>✅ Calibration Complete!</strong><br>
                <small>${this.calibrationResults.message}</small>
            </div>
        `;
    }
    
    formatMatrix(matrix) {
        return matrix.map(row => 
            '[' + row.map(val => val.toFixed(2).padStart(8)).join(' ') + ']'
        ).join('\\n');
    }
    
    switchView(view) {
        this.currentView = view;
        
        // Update active button
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === view);
        });
        
        // Update view content (implementation depends on available data)
        // For now, just show the original images
        this.displayUploadedImages();
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
            document.getElementById('results-grid').innerHTML = `
                <div class="placeholder-message">
                    <div class="placeholder-icon">📷</div>
                    <h4>Upload Images to Start</h4>
                    <p>Upload chessboard calibration images to see results here</p>
                </div>
            `;
            document.getElementById('calibration-metrics').style.display = 'none';
            document.getElementById('results-summary').innerHTML = '<p>No calibration results yet</p>';
            
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
}

// Initialize the application
let intrinsicCalib;
document.addEventListener('DOMContentLoaded', () => {
    intrinsicCalib = new IntrinsicCalibration();
});
