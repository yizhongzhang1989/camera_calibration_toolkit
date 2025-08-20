/**
 * Intrinsic Camera Calibration Web Interface
 * Handles file uploads, parameter settings, and calibration workflow
 * Uses shared ChessboardConfig module for chessboard functionality
 * Extends BaseCalibration for common functionality
 */

class IntrinsicCalibration extends BaseCalibration {
    constructor() {
        super('intrinsic');
        this.initializeEventListeners();
        this.initializeConsole();
        this.updateUI();
    }
    
    // ========================================
    // Intrinsic-Specific Overrides
    // ========================================
    
    getImageUploadMessage() {
        return 'calibration images';
    }
    
    getDefaultProgressMessage() {
        return 'Upload calibration images to see results';
    }
    
    getGlobalInstanceName() {
        return 'intrinsicCalib';
    }
    
    getCalibrationButtonText() {
        return 'Start Intrinsic Calibration';
    }
    
    // ========================================
    // Event Listeners Setup
    // ========================================
    
    initializeEventListeners() {
        console.log('🔧 Initializing event listeners for intrinsic calibration');
        
        // File upload
        const imageFiles = document.getElementById('image-files');
        const uploadArea = document.getElementById('image-upload-area');
        
        console.log('📁 Image files element:', imageFiles);
        console.log('📂 Upload area element:', uploadArea);
        
        if (imageFiles) {
            imageFiles.addEventListener('change', (e) => {
                console.log('📷 File input changed, files:', e.target.files);
                this.handleImageUpload(e.target.files);
            });
            console.log('✅ Image files event listener added');
        } else {
            console.error('❌ Image files element not found!');
        }
        
        // Drag and drop
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
                console.log('📷 Files dropped:', e.dataTransfer.files);
                this.handleImageUpload(e.dataTransfer.files);
            });
            console.log('✅ Drag and drop event listeners added');
        } else {
            console.error('❌ Upload area element not found!');
        }
        
        // Parameters change listeners
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
        
        // Image actions bar setup - removed (no longer needed)
    }
    
    // ========================================
    // Parameter Management
    // ========================================
    
    async updateParameters() {
        // Get pattern configuration from ChessboardConfig if available
        let patternJSON = null;
        if (window.chessboardConfig && window.chessboardConfig.patternConfigJSON) {
            patternJSON = window.chessboardConfig.getPatternJSON();
        }

        const parameters = {
            session_id: this.sessionId,
            distortion_model: document.getElementById('distortion-model').value
        };

        // Use new JSON pattern format if available
        if (patternJSON) {
            parameters.pattern_json = patternJSON;
            console.log('Using JSON pattern configuration:', patternJSON);
        } else {
            // Fallback to legacy format for backward compatibility
            parameters.chessboard_x = parseInt(document.getElementById('chessboard-x').value);
            parameters.chessboard_y = parseInt(document.getElementById('chessboard-y').value);
            parameters.square_size = parseFloat(document.getElementById('square-size').value);
            console.log('Using legacy parameter format');
        }

        try {
            const response = await fetch('/api/set_parameters', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(parameters)
            });

            const result = await response.json();
            if (result.error) {
                this.showStatus(`Parameter error: ${result.error}`, 'error');
            }

        } catch (error) {
            this.showStatus(`Parameter update failed: ${error.message}`, 'error');
        }
    }    // ========================================
    // Calibration Process
    // ========================================
    
    async startCalibration() {
        const selectedIndices = this.getSelectedImageIndices();
        
        if (selectedIndices.length < 3) {
            this.showStatus('Need at least 3 selected images for calibration', 'error');
            return;
        }
        
        try {
            this.showStatus(`Starting intrinsic calibration with ${selectedIndices.length} selected images...`, 'info');
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
                    calibration_type: 'intrinsic',
                    selected_indices: selectedIndices
                })
            });
            
            const result = await response.json();
            
            if (result.error) {
                this.showStatus(`Calibration failed: ${result.error}`, 'error');
                if (calibrateBtn) calibrateBtn.disabled = false;
                this.stopConsolePolling();
                return;
            }
            
            if (!result.success) {
                this.showStatus('Calibration failed: Unknown error', 'error');
                if (calibrateBtn) calibrateBtn.disabled = false;
                this.stopConsolePolling();
                return;
            }
            
            this.calibrationResults = result;
            this.showStatus(result.message, 'success');
            this.displayResults();
            this.updateUI();
            
            // Stop console polling after successful calibration
            this.stopConsolePolling();
            // Do one final refresh to get any remaining messages
            setTimeout(() => this.refreshConsole(), 500);
            
        } catch (error) {
            this.showStatus(`Calibration failed: ${error.message}`, 'error');
            const calibrateBtn = document.getElementById('calibrate-btn');
            if (calibrateBtn) calibrateBtn.disabled = false;
            this.stopConsolePolling();
        }
    }
    
    // ========================================
    // Results Display
    // ========================================
    
    displayResults() {
        if (!this.calibrationResults || !this.calibrationResults.success) return;
        
        // Show metrics in control panel
        const metricsDiv = document.getElementById('calibration-metrics');
        if (metricsDiv) {
            metricsDiv.style.display = 'block';
            
            // Update metric displays with proper type checking
            if (this.calibrationResults.camera_matrix) {
                this.updateMetricDisplay('camera-matrix-display', this.formatMatrix(this.calibrationResults.camera_matrix));
            }
            if (this.calibrationResults.distortion_coefficients) {
                const distortionModel = this.calibrationResults.distortion_model || 'standard';
                this.updateMetricDisplay('distortion-coeffs-display', this.formatDistortionCoeffs(this.calibrationResults.distortion_coefficients, distortionModel));
            }
            if (typeof this.calibrationResults.rms_error === 'number') {
                this.updateMetricDisplay('rms-error-display', `<strong>${this.calibrationResults.rms_error.toFixed(4)} pixels</strong>`);
            }
            if (typeof this.calibrationResults.images_used === 'number') {
                this.updateMetricDisplay('images-used-display', `<strong>${this.calibrationResults.images_used} images</strong>`);
            }
            
            // Update progress info
            const progressInfo = document.getElementById('progress-info');
            if (progressInfo) {
                progressInfo.innerHTML = `✅ Calibration complete with ${this.calibrationResults.images_used} images`;
            }
        }
        
        // Update table with corner detection and undistorted images
        this.updateImageTable();
    }
    
    updateMetricDisplay(elementId, content) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = content;
        }
    }
    
    formatMatrix(matrix) {
        if (!Array.isArray(matrix)) return '<pre>Invalid matrix data</pre>';
        
        return '<pre>' + matrix.map(row => {
            if (!Array.isArray(row)) return '[Invalid row]';
            return '[' + row.map(val => {
                const num = parseFloat(val);
                return isNaN(num) ? 'NaN' : num.toFixed(4).padStart(10);
            }).join('  ') + ']';
        }).join('\n') + '</pre>';
    }
    
    formatDistortionCoeffs(coeffs, distortionModel = 'standard') {
        if (!Array.isArray(coeffs)) return '<pre>Invalid coefficients data</pre>';
        
        // Handle both 1D and 2D arrays (OpenCV sometimes returns 2D)
        let flatCoeffs = coeffs;
        if (coeffs.length > 0 && Array.isArray(coeffs[0])) {
            // 2D array - flatten it
            flatCoeffs = coeffs[0];
        }
        
        if (!Array.isArray(flatCoeffs)) return '<pre>Invalid coefficients format</pre>';
        
        // Determine expected coefficient count based on distortion model
        const expectedCoeffCounts = {
            'standard': 5,    // k1, k2, p1, p2, k3
            'rational': 8,    // k1-k6, p1, p2  
            'thin_prism': 12, // k1-k6, p1, p2, s1-s4
            'tilted': 14      // k1-k6, p1, p2, s1-s4, τx, τy
        };
        
        const expectedCount = expectedCoeffCounts[distortionModel] || 5;
        
        // Remove trailing zeros beyond the expected count
        let displayCoeffs = [...flatCoeffs];
        
        // For rational model (8 coeffs), if we have more than 8 coefficients,
        // check if coefficients beyond index 7 are all zeros and remove them
        if (distortionModel === 'rational' && displayCoeffs.length > 8) {
            let hasNonZeroAfter8 = false;
            for (let i = 8; i < displayCoeffs.length; i++) {
                if (Math.abs(parseFloat(displayCoeffs[i])) > 1e-10) {
                    hasNonZeroAfter8 = true;
                    break;
                }
            }
            if (!hasNonZeroAfter8) {
                displayCoeffs = displayCoeffs.slice(0, 8);
            }
        }
        
        // General trailing zero removal for all models
        while (displayCoeffs.length > expectedCount && 
               Math.abs(parseFloat(displayCoeffs[displayCoeffs.length - 1])) < 1e-10) {
            displayCoeffs.pop();
        }
        
        return '<pre>[' + displayCoeffs.map((val, index) => {
            const num = parseFloat(val);
            const formattedNum = isNaN(num) ? 'NaN' : num.toFixed(6);
            return `  ${formattedNum}`;
        }).join(',\n') + '\n]</pre>';
    }
    
    
    // ========================================
    // Image Actions and Selection Management - Simplified
    // ========================================
    
    // Override displayUploadedImages to setup selection events after images are added
    displayUploadedImages() {
        super.displayUploadedImages();
        
        // Set up event listeners after images are displayed
        setTimeout(() => {
            this.setupImageTableEventListeners();
        }, 50);
    }
    
    // Override to add selected count updates
    setupImageTableEventListeners() {
        // Call parent implementation first
        if (super.setupImageTableEventListeners) {
            super.setupImageTableEventListeners();
        }
        
        // Add select all functionality with count updates
        const selectAllCheckbox = document.getElementById('select-all-checkbox');
        if (selectAllCheckbox) {
            selectAllCheckbox.addEventListener('change', (e) => {
                const imageCheckboxes = document.querySelectorAll('.image-checkbox');
                imageCheckboxes.forEach(checkbox => {
                    checkbox.checked = e.target.checked;
                });
                this.updateSelectedCount();
            });
        }
        
        // Add individual checkbox listeners with count updates
        const imageCheckboxes = document.querySelectorAll('.image-checkbox');
        imageCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateSelectAllState();
                this.updateSelectedCount();
            });
        });
    }

    updateImageTable() {
        if (!this.calibrationResults) return;
        
        const cornerImages = this.calibrationResults.corner_images || [];
        const undistortedImages = this.calibrationResults.undistorted_images || [];
        
        // Create lookup maps by index (skip null entries)
        const cornerImageMap = {};
        const undistortedImageMap = {};
        
        cornerImages.forEach((img, idx) => {
            if (img !== null && img !== undefined) {
                cornerImageMap[img.index] = img;
            }
        });
        
        undistortedImages.forEach((img, idx) => {
            if (img !== null && img !== undefined) {
                undistortedImageMap[img.index] = img;
            }
        });
        
        this.uploadedImages.forEach((file, index) => {
            // Update corner detection column
            const cornerCell = document.getElementById(`corner-cell-${index}`);
            if (cornerCell) {
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
            }
            
            // Update undistorted column
            const undistortedCell = document.getElementById(`undistorted-cell-${index}`);
            if (undistortedCell) {
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
            }
        });
    }
    
    // ========================================
    // View Switching
    // ========================================
    
    switchView(view) {
        this.currentView = view;
        
        // Update active button
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-view="${view}"]`).classList.add('active');
        
        // Show/hide appropriate sections
        const imageView = document.getElementById('image-comparison-view');
        const metricsView = document.getElementById('metrics-view');
        
        if (imageView && metricsView) {
            if (view === 'images') {
                imageView.style.display = 'block';
                metricsView.style.display = 'none';
            } else if (view === 'metrics') {
                imageView.style.display = 'none';
                metricsView.style.display = 'block';
            }
        }
    }
    
    // ========================================
    // UI State Management Overrides
    // ========================================
    
    resetUIAfterClear() {
        super.resetUIAfterClear();
        
        // Reset metrics view
        const metricsDiv = document.getElementById('calibration-metrics');
        if (metricsDiv) {
            metricsDiv.style.display = 'none';
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
            consoleOutput.textContent = 'Console initialized. Ready for calibration...\n';
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
let intrinsicCalib;
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 DOM Content Loaded - Initializing IntrinsicCalibration');
    try {
        intrinsicCalib = new IntrinsicCalibration();
        console.log('✅ IntrinsicCalibration initialized successfully');
    } catch (error) {
        console.error('❌ Failed to initialize IntrinsicCalibration:', error);
    }
});
