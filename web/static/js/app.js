// Camera Calibration Toolkit - JavaScript Application

class CalibrationApp {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.uploadedImages = [];
        this.uploadedPoses = [];
        this.calibrationResults = null;
        this.currentView = 'original';
        
        this.initializeEventListeners();
        this.initializeThreeJS();
        this.updateUI();
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeEventListeners() {
        // File uploads
        document.getElementById('imageFiles').addEventListener('change', (e) => {
            this.handleImageUpload(e.target.files);
        });
        
        document.getElementById('poseFiles').addEventListener('change', (e) => {
            this.handlePoseUpload(e.target.files);
        });
        
        // Calibration type change
        document.getElementById('calibrationType').addEventListener('change', (e) => {
            this.updateCalibrationType(e.target.value);
        });
        
        // Action buttons
        document.getElementById('calibrateBtn').addEventListener('click', () => {
            this.startCalibration();
        });
        
        document.getElementById('exportBtn').addEventListener('click', () => {
            this.exportResults();
        });
        
        document.getElementById('clearBtn').addEventListener('click', () => {
            this.clearSession();
        });
        
        // View controls
        document.getElementById('showOriginal').addEventListener('click', () => {
            this.changeView('original');
        });
        
        document.getElementById('showCorners').addEventListener('click', () => {
            this.changeView('corners');
        });
        
        document.getElementById('showReprojection').addEventListener('click', () => {
            this.changeView('reprojection');
        });
        
        // Parameter validation
        ['chessboardX', 'chessboardY', 'squareSize'].forEach(id => {
            document.getElementById(id).addEventListener('input', () => {
                this.updateUI();
            });
        });
    }
    
    initializeThreeJS() {
        const container = document.getElementById('threejsContainer');
        if (!container) return;
        
        // Scene setup
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setClearColor(0x000000, 0);
        container.appendChild(this.renderer.domElement);
        
        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1).normalize();
        this.scene.add(directionalLight);
        
        // Default scene
        this.setupDefaultScene();
        
        // Animation loop
        this.animate();
        
        // Handle resize
        window.addEventListener('resize', () => {
            this.onWindowResize();
        });
    }
    
    setupDefaultScene() {
        // Add coordinate axes
        const axesHelper = new THREE.AxesHelper(0.5);
        this.scene.add(axesHelper);
        
        // Add a simple chessboard representation
        const geometry = new THREE.PlaneGeometry(0.4, 0.3);
        const material = new THREE.MeshBasicMaterial({ 
            color: 0x888888,
            transparent: true,
            opacity: 0.7
        });
        const plane = new THREE.Mesh(geometry, material);
        plane.rotation.x = -Math.PI / 2;
        this.scene.add(plane);
        
        // Position camera
        this.camera.position.set(1, 1, 1);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        const container = document.getElementById('threejsContainer');
        if (!container || !this.camera || !this.renderer) return;
        
        this.camera.aspect = container.clientWidth / container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(container.clientWidth, container.clientHeight);
    }
    
    updateCalibrationType(type) {
        const poseUploadGroup = document.getElementById('poseUploadGroup');
        if (type === 'eye_in_hand' || type === 'eye_to_hand') {
            poseUploadGroup.style.display = 'block';
        } else {
            poseUploadGroup.style.display = 'none';
        }
        this.updateUI();
    }
    
    async handleImageUpload(files) {
        if (files.length === 0) return;
        
        this.showLoading('Uploading images...');
        
        const formData = new FormData();
        formData.append('session_id', this.sessionId);
        formData.append('calibration_type', document.getElementById('calibrationType').value);
        
        Array.from(files).forEach((file, index) => {
            formData.append('files', file);
        });
        
        try {
            const response = await fetch('/api/upload_images', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.uploadedImages = result.files;
                this.updateUI();
                this.showStatus(`Uploaded ${result.files.length} images successfully`, 'success');
            } else {
                this.showStatus(`Error: ${result.error}`, 'error');
            }
        } catch (error) {
            this.showStatus(`Error uploading images: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async handlePoseUpload(files) {
        if (files.length === 0) return;
        
        this.showLoading('Uploading pose files...');
        
        const formData = new FormData();
        formData.append('session_id', this.sessionId);
        
        Array.from(files).forEach((file, index) => {
            formData.append('files', file);
        });
        
        try {
            const response = await fetch('/api/upload_poses', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.uploadedPoses = result.files;
                this.updateUI();
                this.showStatus(`Uploaded ${result.files.length} pose files successfully`, 'success');
            } else {
                this.showStatus(`Error: ${result.error}`, 'error');
            }
        } catch (error) {
            this.showStatus(`Error uploading poses: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async startCalibration() {
        // Validate parameters
        const parameters = this.getCalibrationParameters();
        if (!this.validateParameters(parameters)) {
            return;
        }
        
        this.showLoading('Running calibration...');
        this.showStatus('Calibration in progress...', 'processing');
        
        try {
            // Set parameters
            const setParamsResponse = await fetch('/api/set_parameters', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    parameters: parameters
                })
            });
            
            if (!setParamsResponse.ok) {
                const error = await setParamsResponse.json();
                throw new Error(error.error);
            }
            
            // Run calibration
            const calibrateResponse = await fetch('/api/calibrate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });
            
            const result = await calibrateResponse.json();
            
            if (calibrateResponse.ok && result.success) {
                this.calibrationResults = result;
                this.displayResults(result);
                this.showStatus(result.message, 'completed');
                this.updateUI();
                
                // Update 3D visualization if applicable
                if (result.calibration_type === 'eye_in_hand') {
                    this.visualizeEyeInHandResults(result);
                }
            } else {
                this.showStatus(`Calibration failed: ${result.error}`, 'error');
            }
        } catch (error) {
            this.showStatus(`Error during calibration: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async exportResults() {
        if (!this.calibrationResults) {
            this.showStatus('No results to export', 'error');
            return;
        }
        
        try {
            const response = await fetch(`/api/export_results/${this.sessionId}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `calibration_results_${this.sessionId}.zip`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                this.showStatus('Results exported successfully', 'success');
            } else {
                const error = await response.json();
                this.showStatus(`Export failed: ${error.error}`, 'error');
            }
        } catch (error) {
            this.showStatus(`Error exporting results: ${error.message}`, 'error');
        }
    }
    
    async clearSession() {
        if (confirm('Are you sure you want to clear all data and start over?')) {
            try {
                await fetch(`/api/clear_session/${this.sessionId}`, {
                    method: 'POST'
                });
                
                // Reset application state
                this.uploadedImages = [];
                this.uploadedPoses = [];
                this.calibrationResults = null;
                this.sessionId = this.generateSessionId();
                
                // Clear UI
                document.getElementById('imageFiles').value = '';
                document.getElementById('poseFiles').value = '';
                document.getElementById('resultsContainer').innerHTML = `
                    <div class="instructions">
                        <h3>Instructions:</h3>
                        <ol>
                            <li>Select calibration type (Eye-in-Hand for robot-mounted cameras)</li>
                            <li>Set chessboard parameters (corners and square size)</li>
                            <li>Upload calibration images showing the chessboard from different angles</li>
                            <li>For Eye-in-Hand: Upload corresponding robot pose JSON files</li>
                            <li>Click "Start Calibration" to begin the process</li>
                            <li>View results and export calibration data</li>
                        </ol>
                        
                        <h4>Pose File Format:</h4>
                        <pre><code>{
    "end_xyzrpy": {
        "x": 0.5,
        "y": 0.2,
        "z": 0.3,
        "rx": 0.0,
        "ry": 0.0,
        "rz": 1.57
    }
}</code></pre>
                    </div>
                `;
                
                this.updateUI();
                this.showStatus('Session cleared', 'success');
            } catch (error) {
                this.showStatus(`Error clearing session: ${error.message}`, 'error');
            }
        }
    }
    
    getCalibrationParameters() {
        return {
            chessboard_x: parseInt(document.getElementById('chessboardX').value),
            chessboard_y: parseInt(document.getElementById('chessboardY').value),
            square_size: parseFloat(document.getElementById('squareSize').value)
        };
    }
    
    validateParameters(parameters) {
        if (parameters.chessboard_x < 3 || parameters.chessboard_y < 3) {
            this.showStatus('Chessboard must have at least 3x3 corners', 'error');
            return false;
        }
        
        if (parameters.square_size <= 0) {
            this.showStatus('Square size must be greater than 0', 'error');
            return false;
        }
        
        if (this.uploadedImages.length === 0) {
            this.showStatus('Please upload calibration images', 'error');
            return false;
        }
        
        const calibrationType = document.getElementById('calibrationType').value;
        if ((calibrationType === 'eye_in_hand' || calibrationType === 'eye_to_hand') && this.uploadedPoses.length === 0) {
            this.showStatus('Please upload robot pose files for eye-in-hand calibration', 'error');
            return false;
        }
        
        if (this.uploadedPoses.length > 0 && this.uploadedImages.length !== this.uploadedPoses.length) {
            this.showStatus('Number of images must match number of pose files', 'error');
            return false;
        }
        
        return true;
    }
    
    updateUI() {
        // Update file counts
        document.getElementById('imageCount').textContent = `${this.uploadedImages.length} images`;
        document.getElementById('poseCount').textContent = `${this.uploadedPoses.length} poses`;
        
        // Update calibrate button state
        const calibrateBtn = document.getElementById('calibrateBtn');
        const parameters = this.getCalibrationParameters();
        const canCalibrate = this.validateParameters(parameters);
        calibrateBtn.disabled = !canCalibrate;
        
        // Update export button state
        document.getElementById('exportBtn').disabled = !this.calibrationResults;
        
        // Update mean error display
        if (this.calibrationResults && this.calibrationResults.mean_error !== undefined) {
            document.getElementById('meanError').textContent = this.calibrationResults.mean_error.toFixed(4) + ' px';
        } else {
            document.getElementById('meanError').textContent = '-';
        }
    }
    
    changeView(viewType) {
        this.currentView = viewType;
        
        // Update button states
        document.querySelectorAll('.view-controls .btn').forEach(btn => {
            btn.classList.remove('btn-primary');
            btn.classList.add('btn-small');
        });
        
        document.getElementById(`show${viewType.charAt(0).toUpperCase() + viewType.slice(1)}`).classList.add('btn-primary');
        
        // Update displayed images (if results are available)
        if (this.calibrationResults) {
            this.displayResults(this.calibrationResults, viewType);
        }
    }
    
    async displayResults(results, viewType = this.currentView) {
        const container = document.getElementById('resultsContainer');
        container.innerHTML = '';
        
        try {
            // Get detailed results with images
            const detailResponse = await fetch(`/api/get_results/${this.sessionId}`);
            const detailResults = await detailResponse.json();
            
            if (!detailResponse.ok) {
                throw new Error(detailResults.error);
            }
            
            const originalImages = detailResults.original_images || [];
            const vizImages = detailResults.visualization_images || [];
            
            console.log('📊 Display results debug info:');
            console.log(`   Original images: ${originalImages.length}`);
            console.log(`   Visualization images: ${vizImages.length}`);
            vizImages.forEach((img, idx) => {
                console.log(`   VizImg ${idx}: ${img.name} (type: ${img.type})`);
            });
            
            // Create image rows
            for (let i = 0; i < originalImages.length; i++) {
                const imageRow = this.createImageRow(i, originalImages[i], vizImages, viewType, results);
                container.appendChild(imageRow);
            }
            
        } catch (error) {
            container.innerHTML = `<div class="error">Error loading results: ${error.message}</div>`;
        }
    }
    
    createImageRow(index, originalImage, vizImages, viewType, results) {
        const row = document.createElement('div');
        row.className = 'image-row';
        
        // Row header
        const header = document.createElement('div');
        header.className = 'row-header';
        
        const title = document.createElement('h4');
        title.textContent = `View ${index + 1}`;
        header.appendChild(title);
        
        if (results.reprojection_errors && results.reprojection_errors[index] !== undefined) {
            const errorInfo = document.createElement('div');
            errorInfo.className = 'error-info';
            errorInfo.textContent = `Error: ${results.reprojection_errors[index].toFixed(4)} px`;
            header.appendChild(errorInfo);
        }
        
        row.appendChild(header);
        
        // Image columns
        const columns = document.createElement('div');
        columns.className = 'image-columns';
        
        // Original image column
        if (viewType === 'original' || viewType === 'all') {
            const originalColumn = this.createImageColumn('Original', originalImage.data);
            columns.appendChild(originalColumn);
        }
        
        // Corner detection column
        if (viewType === 'corners' || viewType === 'all') {
            const cornerImage = this.findVizImage(vizImages, index, 'corners');
            if (cornerImage) {
                const cornerColumn = this.createImageColumn('Detected Corners', cornerImage.data);
                columns.appendChild(cornerColumn);
            }
        }
        
        // Undistorted axes column  
        if (viewType === 'axes' || viewType === 'all') {
            const axesImage = this.findVizImage(vizImages, index, 'axes');
            if (axesImage) {
                const axesColumn = this.createImageColumn('Undistorted Axes', axesImage.data);
                columns.appendChild(axesColumn);
            }
        }
        
        // Reprojection column
        if (viewType === 'reprojection' || viewType === 'all') {
            const reprojImage = this.findVizImage(vizImages, index, 'reproject');
            if (reprojImage) {
                const reprojColumn = this.createImageColumn('Reprojection', reprojImage.data);
                columns.appendChild(reprojColumn);
            }
        }
        
        row.appendChild(columns);
        return row;
    }
    
    createImageColumn(title, imageData) {
        const column = document.createElement('div');
        column.className = 'image-column';
        
        const titleEl = document.createElement('h5');
        titleEl.textContent = title;
        column.appendChild(titleEl);
        
        const img = document.createElement('img');
        img.src = imageData;
        img.alt = title;
        column.appendChild(img);
        
        return column;
    }
    
    findVizImage(vizImages, index, type) {
        // Map the requested type to the actual types stored in the results
        const typeMapping = {
            'corners': 'corner_detection',
            'reproject': 'reprojection',
            'axes': 'undistorted_axes'
        };
        
        const actualType = typeMapping[type] || type;
        
        // Get the original image name for this index
        const originalImage = this.uploadedImages[index];
        let result = null;
        
        if (originalImage && originalImage.name) {
            const baseName = originalImage.name.replace(/\.[^/.]+$/, ""); // Remove extension
            
            // Look for images that match both the type and the base filename
            result = vizImages.find(img => 
                img.type === actualType && img.name.includes(baseName)
            );
            
            console.log(`🔍 Looking for ${type} (${actualType}) image for ${baseName}: ${result ? '✅ Found ' + result.name : '❌ Not found'}`);
        }
        
        // Fallback: try to find by index and type (for legacy compatibility)
        if (!result) {
            result = vizImages.find(img => 
                img.type === actualType && img.name.includes(`${index}`)
            );
            if (result) {
                console.log(`🔍 Fallback found ${type} image by index ${index}: ${result.name}`);
            }
        }
        
        return result;
    }
    
    visualizeEyeInHandResults(results) {
        if (!this.scene || !results.cam2end_matrix) return;
        
        // Clear existing visualization objects
        const objectsToRemove = [];
        this.scene.traverse((child) => {
            if (child.userData.isCalibrationViz) {
                objectsToRemove.push(child);
            }
        });
        objectsToRemove.forEach(obj => this.scene.remove(obj));
        
        // Add camera coordinate system
        const cameraAxes = new THREE.AxesHelper(0.2);
        cameraAxes.userData.isCalibrationViz = true;
        this.scene.add(cameraAxes);
        
        // Add end-effector coordinate system
        const cam2endMatrix = new THREE.Matrix4();
        const matrixArray = results.cam2end_matrix;
        cam2endMatrix.set(
            matrixArray[0][0], matrixArray[0][1], matrixArray[0][2], matrixArray[0][3],
            matrixArray[1][0], matrixArray[1][1], matrixArray[1][2], matrixArray[1][3],
            matrixArray[2][0], matrixArray[2][1], matrixArray[2][2], matrixArray[2][3],
            matrixArray[3][0], matrixArray[3][1], matrixArray[3][2], matrixArray[3][3]
        );
        
        const endEffectorAxes = new THREE.AxesHelper(0.15);
        endEffectorAxes.applyMatrix4(cam2endMatrix);
        endEffectorAxes.userData.isCalibrationViz = true;
        this.scene.add(endEffectorAxes);
        
        // Add connection line
        const points = [
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(matrixArray[0][3], matrixArray[1][3], matrixArray[2][3])
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({ color: 0x00ff00 });
        const line = new THREE.Line(geometry, material);
        line.userData.isCalibrationViz = true;
        this.scene.add(line);
    }
    
    showLoading(message = 'Processing...') {
        const overlay = document.getElementById('loadingOverlay');
        const text = overlay.querySelector('.loading-text');
        text.textContent = message;
        overlay.style.display = 'flex';
    }
    
    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }
    
    showStatus(message, type = 'info') {
        const statusEl = document.getElementById('calibrationStatus');
        statusEl.textContent = message;
        
        // Remove existing status classes
        statusEl.classList.remove('status-ready', 'status-processing', 'status-completed', 'status-error');
        
        // Add appropriate status class
        switch (type) {
            case 'success':
            case 'completed':
                statusEl.classList.add('status-completed');
                break;
            case 'processing':
                statusEl.classList.add('status-processing');
                break;
            case 'error':
                statusEl.classList.add('status-error');
                break;
            default:
                statusEl.classList.add('status-ready');
        }
        
        // Auto-clear success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                statusEl.textContent = 'Ready';
                statusEl.classList.remove('status-completed');
                statusEl.classList.add('status-ready');
            }, 5000);
        }
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.calibrationApp = new CalibrationApp();
});
