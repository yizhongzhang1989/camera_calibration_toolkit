/**
 * Chessboard Configuration Module
 * Handles dynamic chessboard pattern configuration and display
 */

class ChessboardConfig {
    constructor(options = {}) {
        console.log('ChessboardConfig constructor called with options:', options);
        
        // Default configuration
        this.config = {
            patternType: '',
            parameters: {}
        };
        
        // Available pattern configurations (loaded from API)
        this.patternConfigurations = {};
        
        console.log('ChessboardConfig initialized');
        this.eventListenersInitialized = false;
    }
    
    // Initialize after DOM is ready
    async initialize() {
        console.log('🚀 ChessboardConfig.initialize() called');
        console.log('🌐 Current URL:', window.location.href);
        console.log('📄 DOM ready state:', document.readyState);
        
        try {
            // Load pattern configurations from API
            console.log('📡 Loading pattern configurations from API...');
            await this.loadPatternConfigurations();
            
            if (!this.eventListenersInitialized) {
                console.log('👂 Initializing event listeners...');
                this.initializeEventListeners();
                this.eventListenersInitialized = true;
            }
            
            // Load from session storage if available
            console.log('💾 Loading from session storage...');
            this.loadFromSession();
            
            // Initial display update
            console.log('🖼️ Updating initial display...');
            this.updateDisplay();
            
            console.log('✅ ChessboardConfig initialization completed successfully');
        } catch (error) {
            console.error('❌ Error initializing ChessboardConfig:', error);
        }
    }
    
    // Load pattern configurations from API
    async loadPatternConfigurations() {
        console.log('Loading pattern configurations from API...');
        
        try {
            const response = await fetch('/api/pattern_configurations');
            const data = await response.json();
            
            if (data.success) {
                // Convert array-based parameters to object-based for easier access
                this.patternConfigurations = {};
                for (const [patternType, config] of Object.entries(data.configurations)) {
                    this.patternConfigurations[patternType] = {
                        ...config,
                        parameters: {}
                    };
                    
                    // Convert parameter array to object
                    if (config.parameters && Array.isArray(config.parameters)) {
                        for (const param of config.parameters) {
                            this.patternConfigurations[patternType].parameters[param.name] = param;
                        }
                    }
                }
                
                console.log('Loaded pattern configurations:', this.patternConfigurations);
                
                // Populate pattern type dropdown
                this.populatePatternTypeDropdown();
                
                return true;
            } else {
                throw new Error(data.error || 'Failed to load pattern configurations');
            }
        } catch (error) {
            console.error('Error loading pattern configurations:', error);
            
            // Fallback to minimal configuration
            this.patternConfigurations = {
                standard: {
                    name: 'Standard Chessboard',
                    description: 'Traditional black and white checkerboard',
                    icon: 'bi-grid-3x3-gap',
                    parameters: {
                        width: { type: 'integer', default: 11, min: 3, max: 20, label: 'Corners (Width)' },
                        height: { type: 'integer', default: 8, min: 3, max: 20, label: 'Corners (Height)' },
                        square_size: { type: 'float', default: 0.020, min: 0.001, max: 1.0, step: 0.001, label: 'Square Size (meters)' }
                    }
                }
            };
            
            this.populatePatternTypeDropdown();
            return false;
        }
    }
    
    // Populate pattern type dropdown
    populatePatternTypeDropdown() {
        console.log('🔄 Starting populatePatternTypeDropdown');
        console.log('📊 Available configurations:', this.patternConfigurations);
        
        const selectElement = document.getElementById('pattern-type-select');
        console.log('🎯 Select element found:', selectElement);
        
        if (!selectElement) {
            console.error('❌ Pattern type select element not found!');
            console.log('🔍 Available elements with pattern-type-select:', document.querySelectorAll('#pattern-type-select'));
            return;
        }
        
        console.log('📝 Current select element innerHTML before:', selectElement.innerHTML);
        console.log('🔢 Current options count:', selectElement.children.length);
        
        // Clear existing options except the first one
        while (selectElement.children.length > 1) {
            console.log('🗑️ Removing option:', selectElement.lastChild.textContent);
            selectElement.removeChild(selectElement.lastChild);
        }
        
        // Add options for each pattern type
        for (const [patternType, config] of Object.entries(this.patternConfigurations)) {
            console.log('➕ Adding option:', patternType, '→', config.name);
            const option = document.createElement('option');
            option.value = patternType;
            option.textContent = config.name;
            selectElement.appendChild(option);
            console.log('✅ Option added successfully');
        }
        
        console.log('🎉 Pattern type dropdown populated with', Object.keys(this.patternConfigurations).length, 'options');
        console.log('📝 Final select element innerHTML:', selectElement.innerHTML);
        console.log('🔢 Final options count:', selectElement.children.length);
    }
    
    initializeEventListeners() {
        console.log('Initializing event listeners...');
        
        // Pattern type selection change
        const patternTypeSelect = document.getElementById('pattern-type-select');
        if (patternTypeSelect) {
            patternTypeSelect.addEventListener('change', (e) => {
                this.handlePatternTypeChange(e.target.value);
            });
        }

        // Apply button
        document.addEventListener('click', (e) => {
            if (e.target.matches('.btn-apply-config')) {
                this.applyConfiguration();
            }
            
            // Download button
            if (e.target.matches('#download-pattern-btn')) {
                this.downloadPattern();
            }
        });

        // Modal show event
        document.addEventListener('shown.bs.modal', (e) => {
            if (e.target.id === 'chessboard-config-modal') {
                this.onModalShow();
            }
        });
        
        console.log('Event listeners initialization complete');
    }
    
    // Handle pattern type change
    handlePatternTypeChange(patternType) {
        console.log('🔄 Pattern type changed to:', patternType);
        console.log('📊 Available configurations:', Object.keys(this.patternConfigurations));
        
        if (!patternType || !this.patternConfigurations[patternType]) {
            console.warn('⚠️ Invalid pattern type or configuration not found');
            this.clearConfigurationForm();
            return;
        }
        
        console.log('✅ Valid pattern type, updating config');
        this.config.patternType = patternType;
        console.log('📋 Pattern config:', this.patternConfigurations[patternType]);
        
        this.generateConfigurationForm(patternType);
        
        console.log('🖼️ Calling updateModalPreview...');
        this.updateModalPreview();
        console.log('✅ handlePatternTypeChange completed');
    }
    
    // Clear configuration form
    clearConfigurationForm() {
        const container = document.getElementById('pattern-config-container');
        if (container) {
            container.innerHTML = `
                <div class="text-muted text-center p-4">
                    <i class="bi bi-arrow-up"></i><br>
                    Select a pattern type to configure its parameters
                </div>
            `;
        }
    }
    
    // Generate configuration form dynamically
    generateConfigurationForm(patternType) {
        console.log('🔧 generateConfigurationForm called for:', patternType);
        const patternConfig = this.patternConfigurations[patternType];
        if (!patternConfig) {
            console.error('❌ No pattern config found for:', patternType);
            return;
        }
        
        const container = document.getElementById('pattern-config-container');
        if (!container) {
            console.error('❌ pattern-config-container not found!');
            return;
        }
        
        let html = `<h6 class="mb-3"><i class="${patternConfig.icon || 'bi-grid-3x3-gap'}"></i> ${patternConfig.name}</h6>`;
        
        // Generate form fields based on parameter configuration
        const parameters = patternConfig.parameters;
        let paramList;
        
        // Handle both array and object formats for parameters
        if (Array.isArray(parameters)) {
            // Convert array format to object format for easier processing
            paramList = parameters;
            console.log('📋 Using array format parameters, count:', paramList.length);
        } else {
            // Convert object format to array
            paramList = Object.keys(parameters).map(key => ({ name: key, ...parameters[key] }));
            console.log('📋 Using object format parameters, count:', paramList.length);
        }
        
        // Organize into rows (2 per row for most cases)
        for (let i = 0; i < paramList.length; i += 2) {
            html += '<div class="row">';
            
            // First column
            if (i < paramList.length) {
                const paramConfig = paramList[i];
                html += `<div class="col-md-6">${this.generateParameterField(paramConfig.name, paramConfig)}</div>`;
            }
            
            // Second column (if exists)
            if (i + 1 < paramList.length) {
                const paramConfig = paramList[i + 1];
                html += `<div class="col-md-6">${this.generateParameterField(paramConfig.name, paramConfig)}</div>`;
            }
            
            html += '</div>';
        }
        
        container.innerHTML = html;
        console.log('📝 Form HTML generated and set, container innerHTML length:', html.length);
        
        // Set default values and add event listeners
        console.log('🎯 Setting default values...');
        this.setDefaultValues(patternType);
        console.log('🎧 Adding parameter event listeners...');
        this.addParameterEventListeners();
        console.log('✅ generateConfigurationForm completed');
    }
    
    // Generate a single parameter field
    generateParameterField(paramName, paramConfig) {
        const fieldId = `modal-${this.config.patternType}-${paramName.replace(/_/g, '-')}`;
        
        let html = `<div class="mb-3">`;
        html += `<label for="${fieldId}" class="form-label">${paramConfig.label || paramName}</label>`;
        
        if (paramConfig.options && Array.isArray(paramConfig.options)) {
            // Dropdown for options
            html += `<select class="form-select" id="${fieldId}" data-param="${paramName}">`;
            for (const option of paramConfig.options) {
                const selected = option.value == paramConfig.default ? 'selected' : '';
                html += `<option value="${option.value}" ${selected}>${option.label}</option>`;
            }
            html += '</select>';
        } else if (paramConfig.input_type === 'select' && paramConfig.options) {
            // Handle input_type: select for dropdown elements
            html += `<select class="form-select" id="${fieldId}" data-param="${paramName}">`;
            for (const option of paramConfig.options) {
                const selected = option.value == paramConfig.default ? 'selected' : '';
                html += `<option value="${option.value}" ${selected}>${option.label}</option>`;
            }
            html += '</select>';
        } else {
            // Input field
            const inputType = paramConfig.type === 'integer' ? 'number' : 'number';
            const step = paramConfig.step || (paramConfig.type === 'integer' ? '1' : '0.001');
            const min = paramConfig.min || '';
            const max = paramConfig.max || '';
            
            html += `<input type="${inputType}" class="form-control" id="${fieldId}" 
                     data-param="${paramName}" value="${paramConfig.default}" 
                     step="${step}" min="${min}" max="${max}">`;
        }
        
        if (paramConfig.description) {
            html += `<div class="form-text">${paramConfig.description}</div>`;
        }
        
        html += '</div>';
        return html;
    }
    
    // Set default values for parameters
    setDefaultValues(patternType) {
        const patternConfig = this.patternConfigurations[patternType];
        if (!patternConfig) return;
        
        // Initialize parameters object if it doesn't exist
        if (!this.config.parameters) {
            this.config.parameters = {};
        }
        
        const parameters = patternConfig.parameters;
        
        // Handle both array and object formats for parameters
        if (Array.isArray(parameters)) {
            // Array format: [{ name: "width", default: 8, ... }, ...]
            for (const paramConfig of parameters) {
                if (!(paramConfig.name in this.config.parameters)) {
                    this.config.parameters[paramConfig.name] = paramConfig.default;
                }
            }
        } else {
            // Object format: { width: { default: 8, ... }, ... }
            for (const [paramName, paramConfig] of Object.entries(parameters)) {
                if (!(paramName in this.config.parameters)) {
                    this.config.parameters[paramName] = paramConfig.default;
                }
            }
        }
    }
    
    // Add event listeners to parameter inputs
    addParameterEventListeners() {
        console.log('addParameterEventListeners() called');
        const container = document.getElementById('pattern-config-container');
        if (!container) {
            console.error('pattern-config-container not found');
            return;
        }
        
        const inputs = container.querySelectorAll('input, select');
        console.log(`Found ${inputs.length} parameter inputs`);
        
        inputs.forEach((input, index) => {
            const paramName = input.getAttribute('data-param');
            console.log(`Adding event listener to input ${index}: ${paramName}, element:`, input);
            
            input.addEventListener('input', (e) => {
                console.log(`Parameter changed: ${paramName} = ${e.target.value}`);
                let value = e.target.value;
                
                // Convert to appropriate type - find parameter config
                const patternConfig = this.patternConfigurations[this.config.patternType];
                let paramConfig = null;
                
                if (Array.isArray(patternConfig.parameters)) {
                    // Array format: find by name
                    paramConfig = patternConfig.parameters.find(p => p.name === paramName);
                } else {
                    // Object format
                    paramConfig = patternConfig.parameters[paramName];
                }
                
                if (paramConfig) {
                    if (paramConfig.type === 'integer') {
                        value = parseInt(value);
                    } else if (paramConfig.type === 'float') {
                        value = parseFloat(value);
                    }
                }
                
                console.log(`Setting config parameter ${paramName} = ${value}`);
                this.config.parameters[paramName] = value;
                console.log('Current config:', this.config);
                this.updateModalPreview();
            });
        });
    }
    
    onModalShow() {
        console.log('Modal shown, loading current config');
        this.loadConfigToModal();
        this.updateModalPreview();
    }
    
    loadFromSession() {
        try {
            const saved = sessionStorage.getItem('chessboard_config');
            if (saved) {
                const savedConfig = JSON.parse(saved);
                this.config = { ...this.config, ...savedConfig };
                console.log('Loaded config from session:', this.config);
            }
        } catch (error) {
            console.warn('Could not load chessboard config from session:', error);
        }
    }

    saveToSession() {
        try {
            sessionStorage.setItem('chessboard_config', JSON.stringify(this.config));
        } catch (error) {
            console.warn('Could not save chessboard config to session:', error);
        }
    }
    
    loadConfigToModal() {
        console.log('loadConfigToModal called with config:', this.config);
        
        if (!this.config.patternType) return;
        
        // Set pattern type dropdown
        const patternTypeSelect = document.getElementById('pattern-type-select');
        if (patternTypeSelect) {
            patternTypeSelect.value = this.config.patternType;
            console.log('Set pattern type dropdown to:', this.config.patternType);
        }
        
        // Generate the form for the current pattern type
        this.generateConfigurationForm(this.config.patternType);
        
        // Load parameter values after the form is generated
        if (this.config.parameters) {
            console.log('Loading parameter values:', this.config.parameters);
            for (const [paramName, value] of Object.entries(this.config.parameters)) {
                const fieldId = `modal-${this.config.patternType}-${paramName.replace(/_/g, '-')}`;
                const element = document.getElementById(fieldId);
                if (element) {
                    element.value = value;
                    console.log(`Set ${paramName} field to ${value}`);
                } else {
                    console.warn(`Could not find element for parameter: ${paramName}, fieldId: ${fieldId}`);
                }
            }
        }
    }
    
    applyConfiguration() {
        console.log('🔧 Applying configuration:', this.config);
        
        this.saveToSession();
        console.log('💾 Configuration saved to session');
        
        this.updateDisplay();
        console.log('🖼️ Display updated');

        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('chessboard-config-modal'));
        if (modal) {
            modal.hide();
            console.log('🚪 Modal closed');
        }
    }
    
    updateDisplay() {
        // Update the main chessboard card display
        this.updatePatternImage();
        this.updatePatternDescription();
    }
    
    updatePatternImage() {
        console.log('🖼️ updatePatternImage called');
        console.log('📊 Current config:', this.config);
        
        const img = document.getElementById('chessboard-preview-img');
        console.log('🎯 Preview img element:', img);
        
        if (!img || !this.config.patternType) {
            console.warn('⚠️ Cannot update pattern image - img:', img, 'patternType:', this.config.patternType);
            return;
        }
        
        const params = {
            pattern_json: JSON.stringify(this.getPatternJSON()),
            pixel_per_square: 20,
            border_pixels: 0,
            t: Date.now() // Cache busting
        };
        
        const url = `/api/pattern_image?${new URLSearchParams(params)}`;
        console.log('🔗 Generated URL (with cache-busting):', url);
        
        img.src = url;
        console.log('🔄 Image src set to:', img.src);
        
        img.onload = () => {
            console.log('✅ Main preview image loaded successfully');
        };
        
        img.onerror = () => {
            console.error('❌ Failed to load pattern image:', url);
            img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIEltYWdlPC90ZXh0Pjwvc3ZnPg==';
        };
    }
    
    async updatePatternDescription() {
        const typeDisplay = document.getElementById('pattern-type-display');
        const specsDisplay = document.getElementById('pattern-specs-display');
        
        if (!typeDisplay || !specsDisplay || !this.config.patternType) return;
        
        const patternConfig = this.patternConfigurations[this.config.patternType];
        if (patternConfig) {
            typeDisplay.textContent = patternConfig.name;
            
            // Generate specs display
            let specs = [];
            if (this.config.parameters) {
                for (const [paramName, value] of Object.entries(this.config.parameters)) {
                    const paramConfig = patternConfig.parameters[paramName];
                    if (paramConfig) {
                        specs.push(`${paramConfig.label || paramName}: ${value}`);
                    }
                }
            }
            specsDisplay.textContent = specs.join(', ');
        }
    }
    
    updateModalPreview() {
        console.log('🖼️ updateModalPreview called');
        console.log('📊 Current config:', this.config);
        
        const img = document.getElementById('chessboard-modal-preview');
        console.log('🎯 Modal preview img element:', img);
        
        if (!img || !this.config.patternType) {
            console.warn('⚠️ Cannot update modal preview - img:', img, 'patternType:', this.config.patternType);
            return;
        }
        
        const params = {
            pattern_json: JSON.stringify(this.getPatternJSON()),
            pixel_per_square: 20,
            border_pixels: 0,
            t: Date.now() // Cache busting
        };
        
        const url = `/api/pattern_image?${new URLSearchParams(params)}`;
        console.log('🔗 Generated URL (with cache-busting):', url);
        
        img.src = url;
        console.log('🔄 Image src set to:', img.src);
        
        img.onload = () => {
            console.log('✅ Modal preview image loaded successfully');
        };
        
        img.onerror = () => {
            console.error('❌ Failed to load modal preview image:', url);
        };
    }
    
    // Create a valid JSON pattern object for the backend
    getPatternJSON() {
        if (!this.config.patternType || !this.config.parameters) {
            // Return default standard chessboard pattern
            return {
                pattern_id: 'standard_chessboard',
                name: 'Standard Chessboard',
                description: 'Traditional black and white checkerboard pattern',
                is_planar: true,
                parameters: {
                    width: 11,
                    height: 8,
                    square_size: 0.025
                }
            };
        }

        // Create the base pattern object
        const patternJSON = {
            pattern_id: this.config.patternType === 'standard' ? 'standard_chessboard' : 'charuco_board',
            is_planar: true,
            parameters: {}
        };

        // Set name and description based on pattern type
        if (this.config.patternType === 'standard') {
            patternJSON.name = 'Standard Chessboard';
            patternJSON.description = 'Traditional black and white checkerboard pattern';
        } else if (this.config.patternType === 'charuco') {
            patternJSON.name = 'ChArUco Board';
            patternJSON.description = 'Chessboard with ArUco markers';
        } else {
            patternJSON.name = this.config.patternType;
            patternJSON.description = `${this.config.patternType} calibration pattern`;
        }

        // Copy all parameters from the current configuration
        for (const [paramName, value] of Object.entries(this.config.parameters)) {
            // Ensure numeric values are properly converted
            if (typeof value === 'string' && !isNaN(value)) {
                patternJSON.parameters[paramName] = paramName.includes('_id') ? parseInt(value) : parseFloat(value);
            } else {
                patternJSON.parameters[paramName] = value;
            }
        }

        return patternJSON;
    }
    
    async downloadPattern() {
        if (!this.config.patternType) {
            alert('Please select and configure a pattern type first.');
            return;
        }
        
        console.log('Downloading pattern with config:', this.config);
        
        const params = {
            pattern_json: JSON.stringify(this.getPatternJSON()),
            pixel_per_square: 100,  // Higher resolution for printing
            border_pixels: 50       // Border for printing
        };
        
        const url = `/api/pattern_image?${new URLSearchParams(params)}`;
        
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = `calibration_pattern_${this.config.patternType}.png`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(downloadUrl);
            document.body.removeChild(a);
            
            console.log('Pattern download successful');
        } catch (error) {
            console.error('Error downloading pattern:', error);
            alert('Error downloading pattern. Please try again.');
        }
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChessboardConfig;
}
