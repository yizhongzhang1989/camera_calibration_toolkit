/**
 * Eye-in-Hand Camera Calibration Web Interface
 * Handles file uploads, parameter settings, and calibration workflow
 * Uses shared ChessboardConfig module for chessboard functionality
 */

class EyeInHandCalibration {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.uploadedImages = [];
        this.uploadedPoses = [];
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
        
        // Override saveConfiguration to integrate with eye-in-hand calibration
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
        return 'eye_in_hand_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeEventListeners() {
        // File upload - Images
        const imageFiles = document.getElementById('image-files');
        const uploadArea = document.getElementById('image-upload-area');
        
        imageFiles.addEventListener('change', (e) => this.handleImageUpload(e.target.files));
        
        // Drag and drop - Images
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
        
        // File upload - Poses
        const poseFiles = document.getElementById('pose-files');
        const poseUploadArea = document.getElementById('pose-upload-area');
        
        poseFiles.addEventListener('change', (e) => this.handlePoseUpload(e.target.files));
        
        // Drag and drop - Poses
        poseUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            poseUploadArea.classList.add('dragover');
        });
        
        poseUploadArea.addEventListener('dragleave', () => {
            poseUploadArea.classList.remove('dragover');
        });
        
        poseUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            poseUploadArea.classList.remove('dragover');
            this.handlePoseUpload(e.dataTransfer.files);
        });
        
        // Camera matrix source change
        document.getElementById('camera-matrix-source').addEventListener('change', (e) => {
            const manualSection = document.getElementById('manual-camera-section');
            manualSection.style.display = e.target.value === 'manual' ? 'block' : 'none';
            this.updateParameters();
        });
        
        // Manual camera parameters change
        ['fx', 'fy', 'cx', 'cy', 'distortion-coeffs'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', () => this.updateParameters());
            }
        });
        
        // Parameters change
        ['chessboard-x', 'chessboard-y', 'square-size', 'handeye-method'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', () => this.updateParameters());
            }
        });
        
        // Buttons
        document.getElementById('calibrate-btn').addEventListener('click', () => this.startCalibration());
        document.getElementById('download-btn').addEventListener('click', () => this.downloadResults());
        document.getElementById('clear-all-images-btn').addEventListener('click', () => this.clearAllImages());
        document.getElementById('clear-all-poses-btn').addEventListener('click', () => this.clearAllPoses());
    }
    
    async handleImageUpload(files) {
        const formData = new FormData();
        formData.append('session_id', this.sessionId);
        formData.append('calibration_type', 'eye_in_hand');
        
        for (let file of files) {
            if (file.type.startsWith('image/')) {
                formData.append('files', file);
            }
        }
        
        try {
            this.showStatus('Uploading robot pose images...', 'info');
            
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
            this.updatePosesStatus();
            
        } catch (error) {
            this.showStatus(`Upload failed: ${error.message}`, 'error');
        }
    }
    
    async handlePoseUpload(files) {
        const formData = new FormData();
        formData.append('session_id', this.sessionId);
        
        for (let file of files) {
            if (file.name.endsWith('.json')) {
                formData.append('files', file);
            }
        }
        
        try {
            this.showStatus('Uploading robot pose files...', 'info');
            
            const response = await fetch('/api/upload_poses', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.error) {
                this.showStatus(`Error: ${result.error}`, 'error');
                return;
            }
            
            // Backend returns all poses (existing + new), so just replace the array
            const allPoses = result.files || [];
            this.uploadedPoses = allPoses;
            
            this.showStatus(result.message, 'success');
            this.updateUI();
            this.updatePosesStatus();
            this.displayUploadedImages(); // Refresh to show pose validation
            
        } catch (error) {
            this.showStatus(`Pose upload failed: ${error.message}`, 'error');
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
            <td colspan="4" style="text-align: center; color: #666; font-style: italic; padding: 1rem;">
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
                        <button class="image-delete-btn" onclick="eyeInHandCalib.removeImage(${index})" title="Remove image">&times;</button>
                    </div>
                </td>
                <td>
                    <div class="comparison-image-container">
                        <img src="${file.url}" alt="Original ${index + 1}" class="comparison-image" loading="lazy" onclick="eyeInHandCalib.openModal('${file.url}', 'Original Image ${index + 1}')">
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
                <td id="reprojected-cell-${index}">
                    <div class="image-placeholder">
                        <div class="placeholder-icon">⏳</div>
                        <div class="placeholder-text">Reprojection pending</div>
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
        const cameraSource = document.getElementById('camera-matrix-source').value;
        
        const parameters = {
            session_id: this.sessionId,
            chessboard_x: parseInt(document.getElementById('chessboard-x').value),
            chessboard_y: parseInt(document.getElementById('chessboard-y').value),
            square_size: parseFloat(document.getElementById('square-size').value),
            handeye_method: document.getElementById('handeye-method').value,
            camera_matrix_source: cameraSource,
            distortion_model: 'standard'  // Default for eye-in-hand
        };
        
        // Add manual camera parameters if selected
        if (cameraSource === 'manual') {
            parameters.fx = parseFloat(document.getElementById('fx').value);
            parameters.fy = parseFloat(document.getElementById('fy').value);
            parameters.cx = parseFloat(document.getElementById('cx').value);
            parameters.cy = parseFloat(document.getElementById('cy').value);
            
            const distCoeffsStr = document.getElementById('distortion-coeffs').value;
            parameters.distortion_coefficients = distCoeffsStr.split(',').map(x => parseFloat(x.trim()));
        }
        
        try {
            const response = await fetch('/api/set_parameters', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(parameters)  // Send parameters at root level
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
            this.showStatus('Need at least 3 selected images for eye-in-hand calibration', 'error');
            return;
        }
        
        if (this.uploadedImages.length !== this.uploadedPoses.length) {
            this.showStatus('Number of images must match number of pose files', 'error');
            return;
        }
        
        if (this.uploadedPoses.length === 0) {
            this.showStatus('Robot pose files are required for eye-in-hand calibration', 'error');
            return;
        }
        
        try {
            this.showStatus(`Starting eye-in-hand calibration with ${selectedIndices.length} selected images...`, 'info');
            document.getElementById('calibrate-btn').disabled = true;
            
            // Update parameters first
            await this.updateParameters();
            
            const response = await fetch('/api/calibrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    calibration_type: 'eye_in_hand',
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
        
        // Hand-eye transformation matrix
        const handeyeTransform = this.calibrationResults.handeye_transform;
        document.getElementById('handeye-transform-display').innerHTML = `
            <pre>${this.formatMatrix(handeyeTransform)}</pre>
        `;
        
        // Rotation matrix (top-left 3x3 of transformation matrix)
        const rotationMatrix = handeyeTransform.slice(0, 3).map(row => row.slice(0, 3));
        document.getElementById('rotation-matrix-display').innerHTML = `
            <pre>${this.format3x3Matrix(rotationMatrix)}</pre>
        `;
        
        // Translation vector (first 3 elements of last column)
        const translationVector = handeyeTransform.slice(0, 3).map(row => row[3]);
        document.getElementById('translation-vector-display').innerHTML = `
            <pre>[${translationVector.map(v => v.toFixed(6)).join('\n ')}]</pre>
        `;
        
        // Reprojection error
        const errorDisplay = document.getElementById('error-display');
        if (this.calibrationResults.reprojection_error !== undefined) {
            errorDisplay.innerHTML = `<strong>${this.calibrationResults.reprojection_error.toFixed(4)} pixels</strong>`;
        } else {
            errorDisplay.innerHTML = '<em>Not calculated</em>';
        }
        
        // Update progress info
        const progressInfo = document.getElementById('progress-info');
        progressInfo.innerHTML = `✅ Eye-in-hand calibration complete with ${this.uploadedImages.length} images`;
        
        // Update table with corner detection, undistorted, and reprojected images
        this.updateImageTable();
    }
    
    updateImageTable() {
        if (!this.calibrationResults) return;
        
        const cornerImages = this.calibrationResults.corner_images || [];
        const undistortedImages = this.calibrationResults.undistorted_images || [];
        const reprojectedImages = this.calibrationResults.reprojected_images || [];
        
        // Create lookup maps by index
        const cornerImageMap = {};
        const undistortedImageMap = {};
        const reprojectedImageMap = {};
        
        cornerImages.forEach(img => {
            cornerImageMap[img.index] = img;
        });
        
        undistortedImages.forEach(img => {
            undistortedImageMap[img.index] = img;
        });
        
        reprojectedImages.forEach(img => {
            reprojectedImageMap[img.index] = img;
        });
        
        this.uploadedImages.forEach((file, index) => {
            // Update corner detection column
            const cornerCell = document.getElementById(`corner-cell-${index}`);
            if (cornerImageMap[index]) {
                cornerCell.innerHTML = `
                    <div class="comparison-image-container">
                        <img src="${cornerImageMap[index].url}" alt="Corners ${index + 1}" class="comparison-image" loading="lazy" onclick="eyeInHandCalib.openModal('${cornerImageMap[index].url}', 'Corner Detection ${index + 1}')">
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
                        <img src="${undistortedImageMap[index].url}" alt="Undistorted ${index + 1}" class="comparison-image" loading="lazy" onclick="eyeInHandCalib.openModal('${undistortedImageMap[index].url}', 'Undistorted Image ${index + 1}')">
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
            
            // Update reprojected column
            const reprojectedCell = document.getElementById(`reprojected-cell-${index}`);
            if (reprojectedImageMap[index]) {
                reprojectedCell.innerHTML = `
                    <div class="comparison-image-container">
                        <img src="${reprojectedImageMap[index].url}" alt="Reprojected ${index + 1}" class="comparison-image" loading="lazy" onclick="eyeInHandCalib.openModal('${reprojectedImageMap[index].url}', 'Reprojected Points ${index + 1}')">
                    </div>
                `;
            } else {
                reprojectedCell.innerHTML = `
                    <div style="color: #666; padding: 2rem;">
                        <div style="font-size: 2rem;">⚠️</div>
                        <div>Reprojection failed</div>
                    </div>
                `;
            }
        });
    }
    
    formatMatrix(matrix) {
        if (!matrix || !Array.isArray(matrix)) return 'Matrix not available';
        
        // Handle different matrix formats
        if (matrix.length === 4 && Array.isArray(matrix[0])) {
            // Already in 4x4 format
            return matrix.map(row => 
                '[' + row.map(val => val.toFixed(4).padStart(10)).join('  ') + ']'
            ).join('\n');
        } else if (matrix.length === 16) {
            // Flat array - convert to 4x4
            const mat4x4 = [];
            for (let i = 0; i < 4; i++) {
                mat4x4.push(matrix.slice(i * 4, (i + 1) * 4));
            }
            return this.formatMatrix(mat4x4);
        } else {
            return 'Invalid matrix format';
        }
    }
    
    format3x3Matrix(matrix) {
        return matrix.map(row => 
            '[' + row.map(val => val.toFixed(4).padStart(10)).join('  ') + ']'
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
    
    async downloadResults() {
        if (!this.calibrationResults) return;
        
        try {
            const response = await fetch(`/api/export_results/${this.sessionId}`);
            const blob = await response.blob();
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `eye_in_hand_calibration_results_${this.sessionId}.zip`;
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
        const hasPoses = this.uploadedPoses.length > 0;
        const hasResults = this.calibrationResults !== null;
        const posesMatch = hasImages && hasPoses && (this.uploadedImages.length === this.uploadedPoses.length);
        
        // Update calibrate button - requires both images and matching poses
        const calibrateBtn = document.getElementById('calibrate-btn');
        if (hasResults) {
            calibrateBtn.textContent = '🔄 Recalibrate';
            calibrateBtn.disabled = !(hasImages && posesMatch);
        } else {
            calibrateBtn.textContent = 'Start Eye-in-Hand Calibration';
            calibrateBtn.disabled = !(hasImages && posesMatch);
        }
        
        // Update button visibility
        document.getElementById('download-btn').style.display = hasResults ? 'block' : 'none';
        document.getElementById('clear-all-images-btn').style.display = hasImages ? 'block' : 'none';
        document.getElementById('clear-all-poses-btn').style.display = hasPoses ? 'block' : 'none';
        
        // Update poses status
        this.updatePosesStatus();
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
            progressInfo.innerHTML = 'Upload robot pose images and poses to see calibration results';
            
            this.updateUI();
            this.updatePosesStatus();
            this.showStatus('All images cleared', 'info');
            
        } catch (error) {
            this.showStatus(`Failed to clear images: ${error.message}`, 'error');
        }
    }
    
    async clearAllPoses() {
        if (this.uploadedPoses.length === 0) return;
        
        if (!confirm(`Are you sure you want to remove all ${this.uploadedPoses.length} pose files?`)) {
            return;
        }
        
        try {
            // Reset local data
            this.uploadedPoses = [];
            
            // Reset UI
            this.updatePosesStatus();
            this.updateUI();
            this.displayUploadedImages(); // Refresh to remove pose validation
            
            this.showStatus('All poses cleared', 'info');
            
        } catch (error) {
            this.showStatus(`Failed to clear poses: ${error.message}`, 'error');
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
    
    updatePosesStatus() {
        const posesStatusSection = document.getElementById('poses-status-section');
        const posesCount = document.getElementById('poses-count');
        const posesMatchStatus = document.getElementById('poses-match-status');
        const posesValidation = document.getElementById('poses-validation');
        
        const hasPoses = this.uploadedPoses.length > 0;
        const hasImages = this.uploadedImages.length > 0;
        
        if (hasPoses) {
            posesStatusSection.style.display = 'block';
            posesCount.textContent = `${this.uploadedPoses.length} poses uploaded`;
            
            if (hasImages) {
                if (this.uploadedImages.length === this.uploadedPoses.length) {
                    posesMatchStatus.textContent = '• Images and poses match';
                    posesMatchStatus.style.color = '#28a745';
                    posesValidation.textContent = '✅ Ready for calibration';
                    posesValidation.style.color = '#28a745';
                } else {
                    posesMatchStatus.textContent = `• Mismatch: ${this.uploadedImages.length} images vs ${this.uploadedPoses.length} poses`;
                    posesMatchStatus.style.color = '#dc3545';
                    posesValidation.textContent = '⚠️ Image-pose count mismatch';
                    posesValidation.style.color = '#dc3545';
                }
            } else {
                posesMatchStatus.textContent = '• Waiting for images';
                posesMatchStatus.style.color = '#6c757d';
                posesValidation.textContent = '';
            }
        } else {
            posesStatusSection.style.display = 'none';
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
let eyeInHandCalib;
document.addEventListener('DOMContentLoaded', () => {
    eyeInHandCalib = new EyeInHandCalibration();
});
