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
        this.initializeConsole();
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
        // Upload mode toggle
        const uploadModeSelect = document.getElementById('upload-mode-select');
        if (uploadModeSelect) {
            uploadModeSelect.addEventListener('change', (e) => this.handleUploadModeChange(e.target.value));
        }
        
        // Paired file upload
        const pairedFiles = document.getElementById('paired-files');
        const pairedUploadArea = document.getElementById('paired-upload-area');
        
        if (pairedFiles) {
            pairedFiles.addEventListener('change', (e) => this.handlePairedUpload(e.target.files));
        }
        
        // Drag and drop - Paired files
        if (pairedUploadArea) {
            pairedUploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                pairedUploadArea.classList.add('dragover');
            });
            
            pairedUploadArea.addEventListener('dragleave', () => {
                pairedUploadArea.classList.remove('dragover');
            });
            
            pairedUploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                pairedUploadArea.classList.remove('dragover');
                this.handlePairedUpload(e.dataTransfer.files);
            });
        }
        
        // File upload - Images (for separate mode)
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
    // Upload Mode Handling
    // ========================================
    
    handleUploadModeChange(mode) {
        const pairedSection = document.getElementById('paired-upload-section');
        const separateSection = document.getElementById('separate-upload-section');
        
        if (mode === 'paired') {
            pairedSection.style.display = 'block';
            separateSection.style.display = 'none';
        } else {
            pairedSection.style.display = 'none';
            separateSection.style.display = 'block';
        }
    }
    
    async handlePairedUpload(files) {
        const formData = new FormData();
        formData.append('session_id', this.sessionId);
        formData.append('calibration_type', this.calibrationType);
        
        // Filter and add files
        const imageFiles = [];
        const jsonFiles = [];
        
        for (let file of files) {
            if (file.type.startsWith('image/') || file.name.match(/\.(jpg|jpeg|png|bmp|tiff|tif)$/i)) {
                imageFiles.push(file);
                formData.append('files', file);
            } else if (file.name.endsWith('.json')) {
                jsonFiles.push(file);
                formData.append('files', file);
            }
        }
        
        if (imageFiles.length === 0 || jsonFiles.length === 0) {
            this.showStatus('Please select both image files and JSON files', 'error');
            return;
        }
        
        try {
            this.showStatus(`Uploading ${imageFiles.length} images and ${jsonFiles.length} JSON files...`, 'info');
            
            const response = await fetch('/api/upload_paired_files', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.error) {
                this.showStatus(`Error: ${result.error}`, 'error');
                return;
            }
            
            // Update both images and poses data
            this.uploadedImages = result.images || [];
            this.uploadedPoses = result.poses || [];
            
            this.showStatus(result.message, 'success');
            this.onImagesUploaded();
            this.updateUI();
            this.updatePosesStatus();
            this.displayUploadedImages();
            
        } catch (error) {
            this.showStatus(`Paired upload failed: ${error.message}`, 'error');
        }
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
        if (window.chessboardConfig && window.chessboardConfig.patternConfigJSON) {
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
            
            // Clear console and start polling
            await this.clearConsole();
            this.startConsolePolling();
            
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
                this.stopConsolePolling();
                if (calibrateBtn) calibrateBtn.disabled = false;
                return;
            }
            
            // Stop console polling after successful calibration
            this.stopConsolePolling();
            
            this.calibrationResults = result;
            this.showStatus(result.message, 'success');
            this.displayResults();
            this.updateUI();
            
            setTimeout(() => this.refreshConsole(), 500);
            
        } catch (error) {
            this.showStatus(`Calibration failed: ${error.message}`, 'error');
            this.stopConsolePolling();
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
    
    async updateImageTable() {
        if (!this.calibrationResults) return;
        
        console.log('🖼️  Updating image table...');
        
        try {
            // Fetch detailed results with visualization images
            const response = await fetch(`/api/get_results/${this.sessionId}`);
            const detailResults = await response.json();
            
            if (!response.ok) {
                console.error('❌ Failed to fetch visualization images:', detailResults.error);
                return;
            }
            
            const visualizationImages = detailResults.visualization_images || [];
            console.log(`📊 Received ${visualizationImages.length} visualization images`);
            
            // Create lookup maps by image index and type
            const imagesByType = {
                'corner_detection': {},
                'undistorted_axes': {},
                'reprojection': {}
            };
            
            visualizationImages.forEach(img => {
                if (!imagesByType[img.type]) {
                    imagesByType[img.type] = {};
                }
                // Extract the index from the image name (e.g., "image_000.jpg" -> "000")
                const match = img.name.match(/image_(\d+)\.jpg/);
                if (match) {
                    const imageIndex = parseInt(match[1], 10); // Convert "000" to 0, "001" to 1, etc.
                    imagesByType[img.type][imageIndex] = img;
                    console.log(`📷 Mapped ${img.type}: index ${imageIndex} -> ${img.name}`);
                }
            });
            
            this.uploadedImages.forEach((file, index) => {
                const originalName = file.name;
                console.log(`🔍 Looking for images for ${originalName} (index ${index})`);
                
                // Update corner detection column
                const cornerCell = document.getElementById(`corner-cell-${index}`);
                if (cornerCell) {
                    const cornerImage = imagesByType.corner_detection[index]; // Use index instead of filename
                    if (cornerImage && cornerImage.data) {
                        cornerCell.innerHTML = `
                            <div class="comparison-image-container">
                                <img src="${cornerImage.data}" alt="Corners ${index + 1}" class="comparison-image" loading="lazy" onclick="eyeInHandCalib.openModal('${cornerImage.data}', 'Corner Detection ${index + 1}')">
                            </div>
                        `;
                        console.log(`✅ Updated corner detection for ${originalName} at index ${index}`);
                    } else {
                        cornerCell.innerHTML = `
                            <div style="color: #dc3545; padding: 2rem;">
                                <div style="font-size: 2rem;">❌</div>
                                <div>No corner detection image</div>
                            </div>
                        `;
                        console.log(`❌ No corner detection image for index ${index} (${originalName})`);
                    }
                }
                
                // Update undistorted column
                const undistortedCell = document.getElementById(`undistorted-cell-${index}`);
                if (undistortedCell) {
                    const undistortedImage = imagesByType.undistorted_axes[index]; // Use index instead of filename
                    if (undistortedImage && undistortedImage.data) {
                        undistortedCell.innerHTML = `
                            <div class="comparison-image-container">
                                <img src="${undistortedImage.data}" alt="Undistorted ${index + 1}" class="comparison-image" loading="lazy" onclick="eyeInHandCalib.openModal('${undistortedImage.data}', 'Undistorted Axes ${index + 1}')">
                            </div>
                        `;
                        console.log(`✅ Updated undistorted axes for ${originalName} at index ${index}`);
                    } else {
                        undistortedCell.innerHTML = `
                            <div style="color: #666; padding: 2rem;">
                                <div style="font-size: 2rem;">⚠️</div>
                                <div>Undistortion failed</div>
                            </div>
                        `;
                        console.log(`❌ No undistorted axes image for index ${index} (${originalName})`);
                    }
                }
                
                // Update reprojected column
                const reprojectedCell = document.getElementById(`reprojected-cell-${index}`);
                if (reprojectedCell) {
                    const reprojectedImage = imagesByType.reprojection[index]; // Use index instead of filename
                    if (reprojectedImage && reprojectedImage.data) {
                        reprojectedCell.innerHTML = `
                            <div class="comparison-image-container">
                                <img src="${reprojectedImage.data}" alt="Reprojected ${index + 1}" class="comparison-image" loading="lazy" onclick="eyeInHandCalib.openModal('${reprojectedImage.data}', 'Reprojection Error ${index + 1}')">
                            </div>
                        `;
                        console.log(`✅ Updated reprojection for ${originalName} at index ${index}`);
                    } else {
                        reprojectedCell.innerHTML = `
                            <div style="color: #666; padding: 2rem;">
                                <div style="font-size: 2rem;">⚠️</div>
                                <div>Reprojection failed</div>
                            </div>
                        `;
                        console.log(`❌ No reprojection image for index ${index} (${originalName})`);
                    }
                }
            });
            
            console.log('✅ Image table update completed');
            
        } catch (error) {
            console.error('💥 Failed to update image table:', error);
        }
    }
    
    // ========================================
    // Console Output Management
    // ========================================
    
    initializeConsole() {
        // Set up console refresh polling
        this.consolePolling = null;
        
        // Add initial message to console
        const consoleOutput = document.getElementById('console-output');
        if (consoleOutput) {
            consoleOutput.textContent = 'Console initialized. Ready for eye-in-hand calibration...\n';
        }
        
        // Set up console controls
        const clearBtn = document.getElementById('clear-console-btn');
        const refreshBtn = document.getElementById('refresh-console-btn');
        
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearConsole());
        }
        
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshConsole());
        }
        
        // Initialize console resizer
        this.initializeConsoleResizer();
    }
    
    initializeConsoleResizer() {
        const resizer = document.getElementById('console-resizer');
        const consoleArea = document.getElementById('console-area');
        const resultsPanel = document.querySelector('.intrinsic-results-panel');
        
        if (!resizer || !consoleArea || !resultsPanel) return;
        
        let isResizing = false;
        let startY = 0;
        let startHeight = 0;
        
        resizer.addEventListener('mousedown', (e) => {
            isResizing = true;
            startY = e.clientY;
            startHeight = parseInt(document.defaultView.getComputedStyle(consoleArea).height, 10);
            
            // Prevent text selection during drag
            document.body.style.userSelect = 'none';
            resizer.style.background = '#007acc';
            
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            
            const deltaY = startY - e.clientY; // Reversed because we're resizing from bottom
            const newHeight = Math.max(80, Math.min(400, startHeight + deltaY));
            
            consoleArea.style.height = newHeight + 'px';
        });
        
        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                document.body.style.userSelect = '';
                resizer.style.background = '';
            }
        });
    }
    
    startConsolePolling() {
        this.stopConsolePolling(); // Stop any existing polling
        
        this.consolePolling = setInterval(() => {
            this.refreshConsole();
        }, 1000); // Poll every second during calibration
    }
    
    stopConsolePolling() {
        if (this.consolePolling) {
            clearInterval(this.consolePolling);
            this.consolePolling = null;
        }
    }
    
    async refreshConsole() {
        try {
            const response = await fetch(`/api/console/${this.sessionId}`);
            const data = await response.json();
            
            if (data.console_output) {
                const consoleOutput = document.getElementById('console-output');
                if (consoleOutput) {
                    consoleOutput.textContent = data.console_output.join('\n');
                    // Auto-scroll to bottom
                    consoleOutput.scrollTop = consoleOutput.scrollHeight;
                }
            }
        } catch (error) {
            console.error('Error fetching console output:', error);
        }
    }
    
    async clearConsole() {
        try {
            await fetch(`/api/console/${this.sessionId}/clear`, { method: 'GET' });
            const consoleOutput = document.getElementById('console-output');
            if (consoleOutput) {
                consoleOutput.textContent = '';
            }
        } catch (error) {
            console.error('Error clearing console:', error);
        }
    }
}

// Initialize the application
let eyeInHandCalib;
document.addEventListener('DOMContentLoaded', () => {
    eyeInHandCalib = new EyeInHandCalibration();
});
