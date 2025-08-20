/**
 * Dynamic Pattern Selection Modal System - v2.0.1 (2025-08-20)
 * ====================================== 
 * 
 * Integrates with existing calibration UI to provide dynamic pattern selection
 * based on the auto-discovery system from the modular pattern architecture.
 * UPDATED: Added horizontal form layout (label + input on same row)
 */

console.log('🎨 PatternModal v2.0.1 loaded with horizontal layout support');

class PatternSelectionModal {
    constructor() {
        this.availablePatterns = {};
        this.selectedPattern = null;
        this.selectedConfig = null;
        this.initialized = false;
        
        // Bind methods to preserve context
        this.loadPatterns = this.loadPatterns.bind(this);
        this.onPatternSelected = this.onPatternSelected.bind(this);
        this.generateParameterForm = this.generateParameterForm.bind(this);
        this.updatePreview = this.updatePreview.bind(this);
        this.applyConfiguration = this.applyConfiguration.bind(this);
    }

    /**
     * Initialize the pattern selection modal
     */
    async initialize() {
        console.log('🎯 Initializing Pattern Selection Modal');
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Load patterns when modal is opened
        const modal = document.getElementById('chessboard-config-modal');
        if (modal) {
            modal.addEventListener('shown.bs.modal', this.loadPatterns);
        }
        
        this.initialized = true;
        console.log('✅ Pattern Selection Modal initialized');
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Pattern type selection
        const patternSelect = document.getElementById('pattern-type-select');
        if (patternSelect) {
            patternSelect.addEventListener('change', this.onPatternSelected);
        }

        // Apply configuration button
        const applyBtn = document.querySelector('.btn-apply-config');
        if (applyBtn) {
            applyBtn.addEventListener('click', this.applyConfiguration);
        }

        // Download pattern button
        const downloadBtn = document.getElementById('download-pattern-btn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', this.downloadPattern);
        }

        // Download JSON button
        const downloadJsonBtn = document.getElementById('download-json-btn');
        if (downloadJsonBtn) {
            downloadJsonBtn.addEventListener('click', this.downloadPatternJson);
        }

        // Load JSON button
        const loadJsonBtn = document.getElementById('load-json-btn');
        if (loadJsonBtn) {
            loadJsonBtn.addEventListener('click', this.loadPatternJson.bind(this));
        }

        // File input for JSON loading
        const jsonFileInput = document.getElementById('json-file-input');
        if (jsonFileInput) {
            jsonFileInput.addEventListener('change', this.handleJsonFileLoad.bind(this));
        }
    }

    /**
     * Load available patterns from the API
     */
    async loadPatterns() {
        if (this.initialized && Object.keys(this.availablePatterns).length > 0) {
            // Already loaded
            return;
        }

        console.log('🔍 Loading available patterns from API...');
        
        const patternSelect = document.getElementById('pattern-type-select');
        if (!patternSelect) return;

        try {
            // Show loading state
            patternSelect.innerHTML = '<option value="">🔄 Loading patterns...</option>';
            patternSelect.disabled = true;

            const response = await fetch('/api/pattern_configurations');
            const data = await response.json();

            if (data.success) {
                this.availablePatterns = data.configurations;
                this.populatePatternSelect();
                console.log(`✅ Loaded ${Object.keys(this.availablePatterns).length} patterns`);
            } else {
                throw new Error(data.error || 'Failed to load pattern configurations');
            }
        } catch (error) {
            console.error('❌ Failed to load patterns:', error);
            patternSelect.innerHTML = '<option value="">❌ Failed to load patterns</option>';
            this.showError('Failed to load pattern configurations: ' + error.message);
        } finally {
            patternSelect.disabled = false;
        }
    }

    /**
     * Populate the pattern selection dropdown
     */
    populatePatternSelect() {
        const patternSelect = document.getElementById('pattern-type-select');
        if (!patternSelect) return;

        // Store current selection from multiple sources for reliability
        let currentSelection = patternSelect.value;
        
        // Try to get current selection from global chessboard config (more reliable)
        if (window.chessboardConfig && window.chessboardConfig.patternConfigJSON && window.chessboardConfig.patternConfigJSON.pattern_id) {
            currentSelection = window.chessboardConfig.patternConfigJSON.pattern_id;
            console.log(`📋 Got current selection from global config JSON: ${currentSelection}`);
        } else {
            console.log(`📋 Current selection from DOM: ${currentSelection}`);
        }

        // Clear existing options (no placeholder option)
        patternSelect.innerHTML = '';
        
        // Determine default pattern - respect current selection if valid
        const patternIds = Object.keys(this.availablePatterns);
        let defaultPatternId = null;
        
        // First priority: use current selection if it's valid
        if (currentSelection && this.availablePatterns[currentSelection]) {
            defaultPatternId = currentSelection;
            console.log(`🎯 Keeping current selection: ${defaultPatternId}`);
        } else {
            // Second priority: use first available pattern
            defaultPatternId = patternIds[0] || null;
            console.log(`🎯 Using first available as default: ${defaultPatternId}`);
        }

        // Add pattern options
        Object.entries(this.availablePatterns).forEach(([patternId, config]) => {
            const option = document.createElement('option');
            option.value = patternId;
            option.textContent = `${config.icon || '📐'} ${config.name}`;
            option.dataset.description = config.description;
            
            // Set as selected if this is the default pattern
            if (patternId === defaultPatternId) {
                option.selected = true;
            }
            
            patternSelect.appendChild(option);
        });
        
        // Only trigger pattern selection if we're changing to a different pattern
        if (defaultPatternId && defaultPatternId !== currentSelection) {
            console.log(`🔄 Changing pattern from ${currentSelection} to ${defaultPatternId}`);
            patternSelect.value = defaultPatternId;
            this.onPatternSelected({ target: { value: defaultPatternId } });
        } else if (defaultPatternId === currentSelection) {
            console.log(`✅ Keeping current pattern selection: ${defaultPatternId}`);
            // Pattern hasn't changed, but we still need to generate the form and enable apply button
            this.selectedPattern = defaultPatternId;
            this.generateParameterForm(); // Generate form even if pattern didn't change
            const applyBtn = document.querySelector('.btn-apply-config');
            if (applyBtn) {
                applyBtn.disabled = false;
                console.log(`🔓 Enabled apply button for existing pattern: ${defaultPatternId}`);
            }
        }

        console.log(`📋 Populated ${Object.keys(this.availablePatterns).length} pattern options`);
        console.log(`🎯 Default pattern selected: ${defaultPatternId}`);
    }

    /**
     * Handle pattern selection
     */
    onPatternSelected(event) {
        const patternId = event.target.value;
        
        if (!patternId) {
            this.clearConfiguration();
            return;
        }

        console.log(`🎯 Selected pattern: ${patternId}`);
        this.selectedPattern = patternId;
        
        // Generate parameter form
        this.generateParameterForm();
        
        // Enable apply button when pattern is selected
        const applyBtn = document.querySelector('.btn-apply-config');
        if (applyBtn) {
            applyBtn.disabled = false;
        }
    }

    /**
     * Generate dynamic parameter configuration form
     */
    generateParameterForm() {
        const container = document.getElementById('pattern-config-container');
        if (!container || !this.selectedPattern) return;

        const config = this.availablePatterns[this.selectedPattern];
        
        // Clear existing content
        container.innerHTML = '';

        // Create form header
        const header = document.createElement('div');
        header.className = 'mb-3';
        header.innerHTML = `
            <h6 class="mb-2">
                <i class="bi bi-gear-fill text-primary"></i> ${config.name} Parameters
            </h6>
            <small class="text-muted">${config.description}</small>
        `;
        container.appendChild(header);

        // Create parameter fields
        const fieldsContainer = document.createElement('div');
        fieldsContainer.className = 'pattern-parameters';

        config.parameters.forEach(param => {
            const field = this.createParameterField(param);
            fieldsContainer.appendChild(field);
        });

        container.appendChild(fieldsContainer);

        // Initialize with default values and update preview
        this.updatePreview();
    }

    /**
     * Create a parameter input field
     */
    createParameterField(param) {
        const fieldDiv = document.createElement('div');
        fieldDiv.className = 'mb-3';

        let inputElement = '';
        const inputId = `modal-param-${param.name}`;

        switch (param.type) {
            case 'integer':
                // Check if this should be a select dropdown instead
                if (param.input_type === 'select' && param.options) {
                    const options = param.options.map(opt => 
                        `<option value="${opt.value}" ${opt.value === param.default ? 'selected' : ''}>${opt.label}</option>`
                    ).join('');
                    inputElement = `
                        <select class="form-select form-select-sm" id="${inputId}" name="${param.name}" required>
                            ${options}
                        </select>
                    `;
                } else {
                    inputElement = `
                        <input type="number" 
                               class="form-control form-control-sm" 
                               id="${inputId}" 
                               name="${param.name}"
                               value="${param.default || ''}"
                               min="${param.min || ''}"
                               max="${param.max || ''}"
                               step="1"
                               required>
                    `;
                }
                break;

            case 'float':
                inputElement = `
                    <input type="number" 
                           class="form-control form-control-sm" 
                           id="${inputId}" 
                           name="${param.name}"
                           value="${param.default || ''}"
                           min="${param.min || ''}"
                           max="${param.max || ''}"
                           step="${param.step || '0.001'}"
                           required>
                `;
                break;

            case 'select':
                const options = param.options.map(opt => 
                    `<option value="${opt.value}" ${opt.value === param.default ? 'selected' : ''}>${opt.label}</option>`
                ).join('');
                inputElement = `
                    <select class="form-select form-select-sm" id="${inputId}" name="${param.name}" required>
                        ${options}
                    </select>
                `;
                break;

            default:
                inputElement = `
                    <input type="text" 
                           class="form-control form-control-sm" 
                           id="${inputId}" 
                           name="${param.name}"
                           value="${param.default || ''}"
                           required>
                `;
        }

        fieldDiv.innerHTML = `
            <div class="row align-items-center">
                <div class="col-sm-5">
                    <label for="${inputId}" class="form-label mb-0">
                        ${param.label}
                        ${param.required !== false ? '<span class="text-danger">*</span>' : ''}
                    </label>
                </div>
                <div class="col-sm-7">
                    ${inputElement}
                </div>
            </div>
            <div class="form-text mt-1 small text-muted">${param.description}</div>
            <div class="invalid-feedback"></div>
        `;

        // Add real-time validation and preview update
        const input = fieldDiv.querySelector('input, select');
        if (input) {
            input.addEventListener('input', () => {
                this.validateField(param, input);
                this.updatePreview();
            });
        }

        return fieldDiv;
    }

    /**
     * Validate a parameter field
     */
    validateField(param, input) {
        let isValid = true;
        let errorMessage = '';

        const value = input.value;

        if (param.type === 'integer') {
            const intValue = parseInt(value);
            if (isNaN(intValue)) {
                isValid = false;
                errorMessage = 'Must be an integer';
            } else if (param.min !== undefined && intValue < param.min) {
                isValid = false;
                errorMessage = `Must be at least ${param.min}`;
            } else if (param.max !== undefined && intValue > param.max) {
                isValid = false;
                errorMessage = `Must be at most ${param.max}`;
            }
        } else if (param.type === 'float') {
            const floatValue = parseFloat(value);
            if (isNaN(floatValue)) {
                isValid = false;
                errorMessage = 'Must be a number';
            } else if (param.min !== undefined && floatValue < param.min) {
                isValid = false;
                errorMessage = `Must be at least ${param.min}`;
            } else if (param.max !== undefined && floatValue > param.max) {
                isValid = false;
                errorMessage = `Must be at most ${param.max}`;
            }
        }

        // Update field validation state
        if (isValid) {
            input.classList.remove('is-invalid');
            // Remove is-valid class to avoid showing green checkmarks
            input.classList.remove('is-valid');
        } else {
            input.classList.remove('is-valid');
            input.classList.add('is-invalid');
            const feedback = input.parentNode.querySelector('.invalid-feedback');
            if (feedback) {
                feedback.textContent = errorMessage;
            }
        }

        return isValid;
    }

    /**
     * Get current parameter configuration as complete JSON
     */
    getCurrentConfiguration() {
        if (!this.selectedPattern) return null;

        const patternInfo = this.availablePatterns[this.selectedPattern];
        if (!patternInfo) return null;

        // Build the complete JSON configuration
        const config = {
            pattern_id: patternInfo.id || this.selectedPattern,
            name: patternInfo.name || this.selectedPattern,
            description: patternInfo.description || `${this.selectedPattern} calibration pattern`,
            is_planar: true,
            parameters: {}
        };

        const container = document.getElementById('pattern-config-container');
        const inputs = container.querySelectorAll('input, select');

        inputs.forEach(input => {
            if (input.name) {
                let value = input.value;
                
                // Convert to appropriate type based on parameter definition
                const param = patternInfo.parameters.find(p => p.name === input.name);
                
                if (param) {
                    if (param.type === 'integer') {
                        value = parseInt(value);
                    } else if (param.type === 'float') {
                        value = parseFloat(value);
                    }
                }
                
                config.parameters[input.name] = value;
            }
        });

        return config;
    }

    /**
     * Update pattern preview
     */
    async updatePreview() {
        const configuration = this.getCurrentConfiguration();
        if (!configuration) return;

        console.log('🖼️ Updating pattern preview...', configuration);

        const previewContainer = document.getElementById('pattern-preview-container');
        const infoDisplay = document.getElementById('pattern-info-display');
        
        if (!previewContainer) return;

        try {
            // Show loading state
            previewContainer.innerHTML = `
                <div class="text-center text-muted p-3">
                    <div class="spinner-border spinner-border-sm" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div><br>
                    <small>Generating preview...</small>
                </div>
            `;

            const response = await fetch('/api/pattern_image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(configuration)
            });

            if (response.ok) {
                const blob = await response.blob();
                const imageUrl = URL.createObjectURL(blob);
                
                previewContainer.innerHTML = `
                    <img src="${imageUrl}" 
                         alt="Pattern Preview" 
                         class="img-fluid border rounded" 
                         style="max-height: 200px;">
                `;

                // Update info display
                if (infoDisplay) {
                    const patternInfo = this.availablePatterns[this.selectedPattern];
                    const paramsList = Object.entries(configuration.parameters)
                        .map(([key, value]) => `<strong>${key}:</strong> ${value}`)
                        .join('<br>');
                    
                    infoDisplay.innerHTML = `
                        <div class="small">
                            <strong>${patternInfo.name}</strong><br>
                            ${paramsList}
                        </div>
                    `;
                }

                // Enable download buttons
                const downloadBtn = document.getElementById('download-pattern-btn');
                if (downloadBtn) {
                    downloadBtn.disabled = false;
                }
                
                const downloadJsonBtn = document.getElementById('download-json-btn');
                if (downloadJsonBtn) {
                    downloadJsonBtn.disabled = false;
                }

                console.log('✅ Preview updated successfully');
            } else {
                throw new Error('Failed to generate preview');
            }
        } catch (error) {
            console.error('❌ Preview update failed:', error);
            previewContainer.innerHTML = `
                <div class="text-center text-danger p-3">
                    <i class="bi bi-exclamation-triangle"></i><br>
                    <small>Preview failed to load</small>
                </div>
            `;
        }
    }

    /**
     * Apply the current configuration to the main UI
     */
    applyConfiguration() {
        const configuration = this.getCurrentConfiguration();
        if (!configuration) {
            this.showError('No configuration to apply');
            return;
        }

        console.log('✅ Applying configuration:', configuration);

        // Store the configuration for use by calibration
        this.selectedConfig = configuration;

        // Update the chessboard config instance with the JSON configuration
        if (window.chessboardConfig) {
            console.log('🔗 Updating global chessboard config with JSON configuration');
            window.chessboardConfig.setPatternConfigJSON(configuration);
        } else {
            console.warn('⚠️ Global chessboard config not found - updating UI only');
        }

        // Update the main UI elements
        this.updateMainUI();

        // Close the modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('chessboard-config-modal'));
        if (modal) {
            modal.hide();
        }

        console.log('✅ Configuration applied successfully');
    }

    /**
     * Update the main UI with selected pattern
     */
    updateMainUI() {
        if (!this.selectedConfig) return;

        const patternInfo = this.availablePatterns[this.selectedPattern];
        
        // Update the pattern display card
        const patternDisplay = document.getElementById('pattern-type-display');
        if (patternDisplay) {
            patternDisplay.textContent = patternInfo.name;
        }

        const specsDisplay = document.getElementById('pattern-specs-display');
        if (specsDisplay) {
            const specs = Object.entries(this.selectedConfig.parameters)
                .map(([key, value]) => `<strong>${this.formatParameterName(key)}:</strong> ${this.formatParameterValue(key, value)}`)
                .join('<br>');
            specsDisplay.innerHTML = specs;
        }

        // Update the preview image in the main card
        const previewImg = document.getElementById('chessboard-preview-img');
        if (previewImg && this.selectedConfig) {
            // Note: Preview update is delegated to chessboard-config.js to avoid conflicts
            console.log('🔄 Preview generation delegated to chessboard-config.js');
        }

        // Set hidden form values for backward compatibility
        this.setLegacyFormValues();

        // Trigger update in main application if it exists
        if (window.app && typeof window.app.updateUI === 'function') {
            window.app.updateUI();
        }
    }

    /**
     * Generate preview for main UI card (disabled to avoid conflicts)
     */
    async generatePreviewForMainUI(imgElement) {
        // Disabled to avoid conflicts with chessboard-config.js
        // The main preview is handled by the legacy chessboard-config.js system
        console.log('🔄 Preview generation delegated to chessboard-config.js');
    }

    /**
     * Set legacy form values for backward compatibility
     */
    setLegacyFormValues() {
        if (!this.selectedConfig) return;

        const params = this.selectedConfig.parameters;

        // Map common parameter names to legacy form fields
        const parameterMap = {
            'width': 'chessboard-x',
            'height': 'chessboard-y', 
            'square_size': 'square-size'
        };

        Object.entries(parameterMap).forEach(([newName, legacyId]) => {
            const element = document.getElementById(legacyId);
            if (element && params[newName] !== undefined) {
                element.value = params[newName];
            }
        });
    }

    /**
     * Format parameter name for display
     */
    formatParameterName(name) {
        return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    /**
     * Format parameter value for display
     */
    formatParameterValue(name, value) {
        if (name.includes('size') && typeof value === 'number') {
            return value + ' m';
        }
        return value;
    }

    /**
     * Download pattern with current settings
     */
    downloadPattern = async () => {
        console.log('🎯 PatternModal: Download button clicked - collecting current form values...');
        
        const configuration = this.getCurrentConfiguration();
        if (!configuration) {
            this.showError('No pattern configuration available');
            return;
        }

        console.log('📊 Current form configuration for download:', configuration);

        // Add download quality and border settings
        const quality = document.getElementById('download-quality')?.value || 'high';
        const border = document.getElementById('download-border')?.value || 'medium';

        // Map quality to pixel_per_square
        const qualityMap = {
            'standard': 100,
            'high': 150,
            'ultra': 200
        };
        const pixel_per_square = qualityMap[quality] || 150;

        // Map border to border_pixels
        const borderMap = {
            'none': 0,
            'small': 25,
            'medium': 50,
            'large': 100
        };
        const border_pixels = borderMap.hasOwnProperty(border) ? borderMap[border] : 50;

        const downloadConfig = {
            ...configuration
        };

        // Build URL with print parameters as query params (API expects them in request.args)
        const apiUrl = new URL('/api/pattern_image', window.location.origin);
        apiUrl.searchParams.set('pixel_per_square', pixel_per_square);
        apiUrl.searchParams.set('border_pixels', border_pixels);

        try {
            console.log('💾 Download Settings Debug:');
            console.log('  - Quality dropdown value:', quality);
            console.log('  - Border dropdown value:', border);
            console.log('  - Mapped pixel_per_square:', pixel_per_square);
            console.log('  - Mapped border_pixels:', border_pixels);
            console.log('  - API URL with query params:', apiUrl.toString());
            
            console.log('💾 Pattern config being sent to API (JSON body):', downloadConfig);
            
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(downloadConfig)
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${this.selectedPattern}_pattern.png`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                console.log('✅ Pattern downloaded successfully');
            } else {
                throw new Error('Download failed');
            }
        } catch (error) {
            console.error('❌ Download failed:', error);
            this.showError('Failed to download pattern: ' + error.message);
        }
    };

    /**
     * Download pattern configuration as JSON
     */
    downloadPatternJson = async () => {
        console.log('📊 PatternModal: JSON download button clicked - collecting current form values...');
        
        const configuration = this.getCurrentConfiguration();
        if (!configuration) {
            this.showError('No pattern configuration available');
            return;
        }

        try {
            // Create complete JSON structure for saving/loading pattern
            const patternJson = {
                pattern_id: this.selectedPattern,
                name: this.availablePatterns[this.selectedPattern]?.name || this.selectedPattern,
                description: this.availablePatterns[this.selectedPattern]?.description || '',
                is_planar: true, // All current patterns are planar
                parameters: configuration.parameters || {}
            };

            console.log('📊 Generated pattern JSON:', patternJson);

            // Create and download JSON file
            const jsonString = JSON.stringify(patternJson, null, 2);
            const blob = new Blob([jsonString], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `${this.selectedPattern}_config.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            console.log('✅ Pattern JSON downloaded successfully');
        } catch (error) {
            console.error('❌ JSON download failed:', error);
            this.showError('Failed to download pattern JSON: ' + error.message);
        }
    };

    /**
     * Load pattern JSON from file
     */
    loadPatternJson = () => {
        console.log('📤 Load JSON button clicked - opening file picker...');
        const fileInput = document.getElementById('json-file-input');
        if (fileInput) {
            fileInput.click();
        }
    };

    /**
     * Handle JSON file load
     */
    handleJsonFileLoad = async (event) => {
        const file = event.target.files[0];
        if (!file) {
            return;
        }

        console.log('📤 Loading JSON file:', file.name);

        try {
            const jsonText = await this.readFileAsText(file);
            const patternConfig = JSON.parse(jsonText);
            
            console.log('📊 Loaded pattern configuration:', patternConfig);

            // Validate the JSON structure
            if (!patternConfig.pattern_id) {
                throw new Error('Invalid JSON: missing pattern_id');
            }

            // Load this configuration into the modal
            await this.loadConfigurationFromJson(patternConfig);

            console.log('✅ Pattern configuration loaded successfully');
            
        } catch (error) {
            console.error('❌ Failed to load JSON:', error);
            this.showError('Failed to load pattern JSON: ' + error.message);
        } finally {
            // Clear the file input so the same file can be selected again
            event.target.value = '';
        }
    };

    /**
     * Read file as text
     */
    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    /**
     * Load configuration from JSON object
     */
    async loadConfigurationFromJson(patternConfig) {
        console.log('🔧 Loading configuration from JSON into modal...');

        // First, ensure patterns are loaded
        await this.loadPatterns();

        // Select the pattern type
        const patternSelect = document.getElementById('pattern-type-select');
        if (patternSelect && patternConfig.pattern_id) {
            patternSelect.value = patternConfig.pattern_id;
            
            // Trigger pattern selection to load the form
            await this.onPatternSelected({
                target: { value: patternConfig.pattern_id }
            });

            // Wait a moment for the form to be generated
            await new Promise(resolve => setTimeout(resolve, 100));

            // Now populate the form with the parameters
            if (patternConfig.parameters) {
                this.populateParameterForm(patternConfig.parameters);
            }

            console.log('✅ Configuration loaded into modal successfully');
        } else {
            throw new Error(`Pattern type '${patternConfig.pattern_id}' not found`);
        }
    }

    /**
     * Populate parameter form with values
     */
    populateParameterForm(parameters) {
        console.log('📝 Populating form with parameters:', parameters);

        for (const [key, value] of Object.entries(parameters)) {
            const input = document.querySelector(`#pattern-config-container input[name="${key}"], #pattern-config-container select[name="${key}"]`);
            if (input) {
                input.value = value;
                console.log(`✓ Set ${key} = ${value}`);
                
                // Trigger change event to update any dependent fields
                input.dispatchEvent(new Event('change', { bubbles: true }));
            } else {
                console.log(`⚠️ Field '${key}' not found in form`);
            }
        }

        // Update preview after loading parameters
        this.updatePreview();
    }

    /**
     * Clear configuration
     */
    clearConfiguration() {
        this.selectedPattern = null;
        this.selectedConfig = null;

        const container = document.getElementById('pattern-config-container');
        if (container) {
            container.innerHTML = `
                <div class="text-muted text-center p-4">
                    <i class="bi bi-arrow-up fs-1"></i><br>
                    <strong>Select a pattern type above</strong><br>
                    <small>Parameters will appear here automatically</small>
                </div>
            `;
        }

        const previewContainer = document.getElementById('pattern-preview-container');
        if (previewContainer) {
            previewContainer.innerHTML = `
                <div class="preview-placeholder text-muted p-4" style="border: 2px dashed #dee2e6; border-radius: 8px;">
                    <i class="bi bi-image fs-1"></i><br>
                    <small>Preview will appear<br>when pattern is configured</small>
                </div>
            `;
        }

        // Disable buttons
        const applyBtn = document.querySelector('.btn-apply-config');
        if (applyBtn) applyBtn.disabled = true;
        
        const downloadBtn = document.getElementById('download-pattern-btn');
        if (downloadBtn) downloadBtn.disabled = true;
        
        const downloadJsonBtn = document.getElementById('download-json-btn');
        if (downloadJsonBtn) downloadJsonBtn.disabled = true;
    }

    /**
     * Show error message
     */
    showError(message) {
        console.error('Pattern Modal Error:', message);
        
        // Create and show Bootstrap alert
        const alertHtml = `
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                <i class="bi bi-exclamation-triangle-fill"></i>
                <strong>Error:</strong> ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        // Insert at top of modal body
        const modalBody = document.querySelector('#chessboard-config-modal .modal-body');
        if (modalBody) {
            modalBody.insertAdjacentHTML('afterbegin', alertHtml);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                const alert = modalBody.querySelector('.alert');
                if (alert) alert.remove();
            }, 5000);
        }
    }

    /**
     * Get the current selected configuration (for external access)
     */
    getSelectedConfiguration() {
        return this.selectedConfig;
    }
}

// Initialize the pattern selection modal when DOM is ready
let patternModal = null;

document.addEventListener('DOMContentLoaded', function() {
    patternModal = new PatternSelectionModal();
    patternModal.initialize();
    
    // Make it globally accessible
    window.patternSelectionModal = patternModal;
    
    console.log('🎯 Pattern Selection Modal system ready');
});
