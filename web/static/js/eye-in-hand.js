/**
 * Eye-in-Hand Camera Calibration Web Interface
 * Handles file uploads, parameter settings, and calibration workflow
 * Uses shared ChessboardConfig module for chessboard functionality
 * Extends BaseCalibration for common functionality
 */

class EyeInHandCalibration extends BaseCalibration {
    constructor() {
        super('eye_in_hand');
        this.uploadedPoses = [];
        this.initializeEventListeners();
        this.updateUI();
    }
    
    // ========================================
    // Eye-in-Hand Specific Overrides
    // ========================================
    
    getImageUploadMessage() {
        return 'robot pose images';
    }
    
    getDefaultProgressMessage() {
        return 'Upload robot pose images and poses to see calibration results';
    }
    
    getGlobalInstanceName() {
        return 'eyeInHandCalib';
    }
    
    getCalibrationButtonText() {
        return 'Start Eye-in-Hand Calibration';
    }
    
    getTableColumnCount() {
        return 5; // selection, original, corner, undistorted, reprojected
    }
    
    getAdditionalColumns(index) {
        // Eye-in-hand specific columns: corner detection, undistorted, reprojected
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
            <td id="reprojected-cell-${index}">
                <div class="image-placeholder">
                    <div class="placeholder-icon">⏳</div>
                    <div class="placeholder-text">Reprojection pending</div>
                </div>
            </td>
        `;
    }
    
    canStartCalibration(hasImages) {
        const hasPoses = this.uploadedPoses.length > 0;
        const posesMatch = hasImages && hasPoses && (this.uploadedImages.length === this.uploadedPoses.length);
        return hasImages && posesMatch;
    }
    
    onImagesUploaded() {
        this.updatePosesStatus();
    }
    
    onImagesChanged() {
        this.updatePosesStatus();
    }
    
    onUIUpdate(hasImages, hasResults) {
        this.updatePosesStatus();
    }
    
    resetUIAfterClear() {
        super.resetUIAfterClear();
        this.updatePosesStatus();
    }
    
    // ========================================
    // Event Listeners Setup
    // ========================================
    
    initializeEventListeners() {
        // File upload - Images
        const imageFiles = document.getElementById('image-files');
        const uploadArea = document.getElementById('image-upload-area');
        
        if (imageFiles) {
            imageFiles.addEventListener('change', (e) => this.handleImageUpload(e.target.files));
        }
        
        // Drag and drop - Images
        if (uploadArea) {
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
        
        // File upload - Poses
        const poseFiles = document.getElementById('pose-files');
        const poseUploadArea = document.getElementById('pose-upload-area');
        
        if (poseFiles) {
            poseFiles.addEventListener('change', (e) => this.handlePoseUpload(e.target.files));
        }
        
        // Drag and drop - Poses
        if (poseUploadArea) {
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
        }
        
        // Camera matrix source change
        const cameraSourceSelect = document.getElementById('camera-matrix-source');
        if (cameraSourceSelect) {
            cameraSourceSelect.addEventListener('change', (e) => {
                const manualSection = document.getElementById('manual-camera-section');
                if (manualSection) {
                    manualSection.style.display = e.target.value === 'manual' ? 'block' : 'none';
                }
                this.updateParameters();
            });
        }
        
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
        const calibrateBtn = document.getElementById('calibrate-btn');
        const downloadBtn = document.getElementById('download-btn');
        const clearImagesBtn = document.getElementById('clear-all-images-btn');
        const clearPosesBtn = document.getElementById('clear-all-poses-btn');
        
        if (calibrateBtn) calibrateBtn.addEventListener('click', () => this.startCalibration());
        if (downloadBtn) downloadBtn.addEventListener('click', () => this.downloadResults());
        if (clearImagesBtn) clearImagesBtn.addEventListener('click', () => this.clearAllImages());
        if (clearPosesBtn) clearPosesBtn.addEventListener('click', () => this.clearAllPoses());
    }
    
    // ========================================
    // Pose File Handling
    // ========================================
    
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
    
    updatePosesStatus() {
        const posesStatusSection = document.getElementById('poses-status-section');
        const posesCount = document.getElementById('poses-count');
        const posesMatchStatus = document.getElementById('poses-match-status');
        const posesValidation = document.getElementById('poses-validation');
        
        if (!posesStatusSection) return;
        
        const hasPoses = this.uploadedPoses.length > 0;
        const hasImages = this.uploadedImages.length > 0;
        
        if (hasPoses) {
            posesStatusSection.style.display = 'block';
            
            if (posesCount) {
                posesCount.textContent = `${this.uploadedPoses.length} poses uploaded`;
            }
            
            if (hasImages) {
                if (this.uploadedImages.length === this.uploadedPoses.length) {
                    if (posesMatchStatus) {
                        posesMatchStatus.textContent = '• Images and poses match';
                        posesMatchStatus.style.color = '#28a745';
                    }
                    if (posesValidation) {
                        posesValidation.textContent = '✅ Ready for calibration';
                        posesValidation.style.color = '#28a745';
                    }
                } else {
                    if (posesMatchStatus) {
                        posesMatchStatus.textContent = `• Mismatch: ${this.uploadedImages.length} images vs ${this.uploadedPoses.length} poses`;
                        posesMatchStatus.style.color = '#dc3545';
                    }
                    if (posesValidation) {
                        posesValidation.textContent = '⚠️ Image-pose count mismatch';
                        posesValidation.style.color = '#dc3545';
                    }
                }
            } else {
                if (posesMatchStatus) {
                    posesMatchStatus.textContent = '• Waiting for images';
                    posesMatchStatus.style.color = '#6c757d';
                }
                if (posesValidation) {
                    posesValidation.textContent = '';
                }
            }
        } else {
            posesStatusSection.style.display = 'none';
        }
        
        // Update clear poses button
        const clearPosesBtn = document.getElementById('clear-all-poses-btn');
        if (clearPosesBtn) {
            clearPosesBtn.style.display = hasPoses ? 'block' : 'none';
        }
    }
    
    // ========================================
    // Parameter Management
    // ========================================
    
    async updateParameters() {
        const cameraSource = document.getElementById('camera-matrix-source')?.value || 'intrinsic';
        
        // Get pattern configuration from ChessboardConfig if available  
        let patternJSON = null;
        if (window.chessboardConfig && window.chessboardConfig.config) {
            patternJSON = window.chessboardConfig.getPatternJSON();
        }
        
        const parameters = {
            session_id: this.sessionId,
            handeye_method: document.getElementById('handeye-method')?.value || 'horaud',
            camera_matrix_source: cameraSource,
            distortion_model: 'standard'  // Default for eye-in-hand
        };

        // Use new JSON pattern format if available
        if (patternJSON) {
            parameters.pattern_json = patternJSON;
            console.log('Using JSON pattern configuration:', patternJSON);
        } else {
            // Fallback to legacy format for backward compatibility
            parameters.chessboard_x = parseInt(document.getElementById('chessboard-x')?.value || 11);
            parameters.chessboard_y = parseInt(document.getElementById('chessboard-y')?.value || 8);
            parameters.square_size = parseFloat(document.getElementById('square-size')?.value || 0.02);
            console.log('Using legacy parameter format');
        }
        
        // Add manual camera parameters if selected
        if (cameraSource === 'manual') {
            parameters.fx = parseFloat(document.getElementById('fx')?.value || 800);
            parameters.fy = parseFloat(document.getElementById('fy')?.value || 800);
            parameters.cx = parseFloat(document.getElementById('cx')?.value || 320);
            parameters.cy = parseFloat(document.getElementById('cy')?.value || 240);
            
            const distCoeffsStr = document.getElementById('distortion-coeffs')?.value || '0,0,0,0,0';
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
    
    // ========================================
    // Calibration Process
    // ========================================
    
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
            const calibrateBtn = document.getElementById('calibrate-btn');
            if (calibrateBtn) calibrateBtn.disabled = true;
            
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
                if (calibrateBtn) calibrateBtn.disabled = false;
                return;
            }
            
            this.calibrationResults = result;
            this.showStatus(result.message, 'success');
            this.displayResults();
            this.updateUI();
            
        } catch (error) {
            this.showStatus(`Calibration failed: ${error.message}`, 'error');
            const calibrateBtn = document.getElementById('calibrate-btn');
            if (calibrateBtn) calibrateBtn.disabled = false;
        }
    }
    
    // ========================================
    // Results Display
    // ========================================
    
    displayResults() {
        if (!this.calibrationResults) return;
        
        // Show metrics in control panel
        const metricsDiv = document.getElementById('calibration-metrics');
        if (metricsDiv) {
            metricsDiv.style.display = 'block';
            
            // Hand-eye transformation matrix
            const handeyeTransform = this.calibrationResults.handeye_transform;
            this.updateMetricDisplay('handeye-transform-display', `<pre>${this.formatMatrix(handeyeTransform)}</pre>`);
            
            // Rotation matrix (top-left 3x3 of transformation matrix)
            const rotationMatrix = handeyeTransform.slice(0, 3).map(row => row.slice(0, 3));
            this.updateMetricDisplay('rotation-matrix-display', `<pre>${this.format3x3Matrix(rotationMatrix)}</pre>`);
            
            // Translation vector (first 3 elements of last column)
            const translationVector = handeyeTransform.slice(0, 3).map(row => row[3]);
            this.updateMetricDisplay('translation-vector-display', `<pre>[${translationVector.map(v => v.toFixed(6)).join('\n ')}]</pre>`);
            
            // Reprojection error
            const errorDisplay = document.getElementById('error-display');
            if (errorDisplay && this.calibrationResults.reprojection_error !== undefined) {
                errorDisplay.innerHTML = `<strong>${this.calibrationResults.reprojection_error.toFixed(4)} pixels</strong>`;
            }
            
            // Update progress info
            const progressInfo = document.getElementById('progress-info');
            if (progressInfo) {
                progressInfo.innerHTML = `✅ Eye-in-hand calibration complete with ${this.uploadedImages.length} images`;
            }
        }
        
        // Update table with corner detection, undistorted, and reprojected images
        this.updateImageTable();
    }
    
    updateMetricDisplay(elementId, content) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = content;
        }
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
            if (cornerCell) {
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
            }
            
            // Update undistorted column
            const undistortedCell = document.getElementById(`undistorted-cell-${index}`);
            if (undistortedCell) {
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
            }
            
            // Update reprojected column
            const reprojectedCell = document.getElementById(`reprojected-cell-${index}`);
            if (reprojectedCell) {
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
            }
        });
    }
}

// Initialize the application
let eyeInHandCalib;
document.addEventListener('DOMContentLoaded', () => {
    eyeInHandCalib = new EyeInHandCalibration();
});
