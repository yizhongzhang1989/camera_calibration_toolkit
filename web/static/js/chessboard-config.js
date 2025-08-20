/**
 * Chessboard Configuration Module - v2.0.1 (2025-08-20)
 * Handles dynamic chessboard pattern configuration and display
 * UPDATED: Added horizontal form layout (label + input on same row)
 */

console.log('🎨 ChessboardConfig v2.0.1 loaded with horizontal layout support');
class ChessboardConfig {
    constructor(options = {}) {
        console.log('ChessboardConfig constructor called with options:', options);
        
        // Store complete pattern configuration as JSON
        this.patternConfigJSON = null;
        
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
            
            // Ensure we have a default pattern type if none is set
            await this.ensureDefaultPattern();
            
            if (!this.eventListenersInitialized) {
                console.log('👂 Initializing event listeners...');
                this.initializeEventListeners();
                this.eventListenersInitialized = true;
            }
            
            // Load from session storage if available
            console.log('💾 Loading from session storage...');
            await this.loadFromSession();
            
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
                
                // Pattern type selection is now handled by the modal
                console.log('📋 Pattern configurations ready for modal-based selection');
                
                return true;
            } else {
                throw new Error(data.error || 'Failed to load pattern configurations');
            }
        } catch (error) {
            console.error('Error loading pattern configurations:', error);
            
            // Show error state instead of hardcoded fallback
            this.patternConfigurations = {};
            this.showAPIError(error);
            return false;
        }
    }

    // Show API error state
    /**
     * Apply dynamic pattern-specific validation based on API metadata
     * This replaces hardcoded validation rules with dynamic ones from the API
     */
    applyDynamicPatternValidation(patternJSON) {
        if (!patternJSON || !patternJSON.pattern_id) return;
        
        const currentPatternConfig = this.patternConfigurations[patternJSON.pattern_id];
        if (!currentPatternConfig) return;

        // Get pattern metadata for validation rules
        const patternMetadata = currentPatternConfig.metadata || {};
        const validationRules = patternMetadata.validation_rules || {};

        console.log(`🔍 Applying dynamic validation for ${patternJSON.pattern_id}:`, validationRules);

        // Apply parameter relationship constraints
        if (validationRules.parameter_relationships) {
            for (const relationship of validationRules.parameter_relationships) {
                this.applyParameterRelationship(patternJSON, relationship);
            }
        }

        // Apply parameter range constraints
        if (validationRules.parameter_ranges) {
            for (const [paramName, range] of Object.entries(validationRules.parameter_ranges)) {
                this.applyParameterRange(patternJSON, paramName, range);
            }
        }

        // Apply default value corrections
        if (validationRules.default_corrections) {
            for (const [paramName, defaultValue] of Object.entries(validationRules.default_corrections)) {
                this.applyDefaultCorrection(patternJSON, paramName, defaultValue);
            }
        }
    }

    /**
     * Apply parameter relationship constraints (e.g., square_size > marker_size)
     */
    applyParameterRelationship(patternJSON, relationship) {
        const { param1, param2, constraint, fix_values } = relationship;
        
        if (!patternJSON.parameters[param1] || !patternJSON.parameters[param2]) return;

        const value1 = patternJSON.parameters[param1];
        const value2 = patternJSON.parameters[param2];

        let constraintMet = false;
        switch (constraint) {
            case 'greater_than':
                constraintMet = value1 > value2;
                break;
            case 'less_than':
                constraintMet = value1 < value2;
                break;
            case 'equal':
                constraintMet = value1 === value2;
                break;
        }

        if (!constraintMet && fix_values) {
            console.warn(`⚠️ Parameter relationship violation: ${param1} ${constraint} ${param2}, applying fix...`);
            if (fix_values[param1] !== undefined) {
                patternJSON.parameters[param1] = fix_values[param1];
            }
            if (fix_values[param2] !== undefined) {
                patternJSON.parameters[param2] = fix_values[param2];
            }
        }
    }

    /**
     * Apply parameter range constraints
     */
    applyParameterRange(patternJSON, paramName, range) {
        if (!patternJSON.parameters[paramName]) return;

        const value = patternJSON.parameters[paramName];
        const { min, max, default_value } = range;

        if ((min !== undefined && value < min) || (max !== undefined && value > max)) {
            console.warn(`⚠️ Parameter ${paramName} out of range [${min}, ${max}], using default: ${default_value}`);
            patternJSON.parameters[paramName] = default_value;
        }
    }

    /**
     * Apply default value corrections for invalid parameters
     */
    applyDefaultCorrection(patternJSON, paramName, defaultValue) {
        if (!patternJSON.parameters[paramName] || patternJSON.parameters[paramName] < 0) {
            console.warn(`⚠️ Invalid ${paramName}, using default: ${defaultValue}`);
            patternJSON.parameters[paramName] = defaultValue;
        }
    }

    showAPIError(error, context = 'API operation') {
        console.error('🚨 API Error - showing error state');
        
        const selectElement = document.getElementById('pattern-type-select');
        if (selectElement) {
            selectElement.innerHTML = '<option value="">❌ Failed to load patterns</option>';
            selectElement.disabled = true;
        }
        
        const errorMsg = document.createElement('div');
        errorMsg.className = 'alert alert-danger mt-2';
        errorMsg.innerHTML = `
            <strong>Pattern Loading Failed:</strong> ${error.message}<br>
            <small>Please refresh the page or check your connection.</small>
        `;
        
        const container = selectElement?.parentNode;
        if (container && !container.querySelector('.alert-danger')) {
            container.appendChild(errorMsg);
        }
    }
    
    async ensureDefaultPattern() {
        console.log('🔍 Ensuring default pattern JSON is set...');
        
        // If we already have a pattern configuration JSON, we're good
        if (this.patternConfigJSON && typeof this.patternConfigJSON === 'object') {
            console.log('✅ Pattern configuration JSON already set:', this.patternConfigJSON);
            return;
        }
        
        console.log('📥 No pattern configuration found, loading default...');
        await this.loadDefaultPatternJSON();
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
        
        // Clear ALL existing options (including placeholder)
        selectElement.innerHTML = '';
        
        // Determine which pattern should be selected by default
        let defaultPatternType = null;
        const patternTypes = Object.keys(this.patternConfigurations);
        
        if (this.config.patternType && this.patternConfigurations[this.config.patternType]) {
            // Use current pattern type if it exists
            defaultPatternType = this.config.patternType;
            console.log('🎯 Using current pattern type as default:', defaultPatternType);
        } else if (patternTypes.length > 0) {
            // Use first available pattern as default
            defaultPatternType = patternTypes[0];
            console.log('🎯 Using first available pattern as default:', defaultPatternType);
        }
        
        // Add options for each pattern type
        for (const [patternType, config] of Object.entries(this.patternConfigurations)) {
            console.log('➕ Adding option:', patternType, '→', config.name);
            const option = document.createElement('option');
            option.value = patternType;
            option.textContent = config.name;
            
            // Mark default pattern as selected
            if (patternType === defaultPatternType) {
                option.selected = true;
                console.log('✅ Set as selected:', patternType);
            }
            
            selectElement.appendChild(option);
            console.log('✅ Option added successfully');
        }
        
        // Update config to match the selected pattern
        if (defaultPatternType) {
            this.config.patternType = defaultPatternType;
            console.log('🔄 Updated config.patternType to:', defaultPatternType);
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
            
            // Download button - PatternModal now handles all downloads
            if (e.target.matches('#download-pattern-btn')) {
                console.log('🛑 ChessboardConfig: Download handled by PatternModal system');
                return; // PatternModal handles all downloads now
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
        
        console.log('✅ Valid pattern type, creating default JSON config');
        
        // Create a default JSON configuration for this pattern type
        const patternConfig = this.patternConfigurations[patternType];
        const defaultParameters = {};
        
        // Build default parameters
        if (patternConfig.parameters) {
            for (const [paramName, paramInfo] of Object.entries(patternConfig.parameters)) {
                defaultParameters[paramName] = paramInfo.default || 0;
            }
        }
        
        // Set the JSON configuration
        this.patternConfigJSON = {
            pattern_id: patternConfig.id || patternType,
            name: patternConfig.name || patternType,
            description: patternConfig.description || `${patternType} calibration pattern`,
            is_planar: true,
            parameters: defaultParameters
        };
        
        console.log('📋 Created default pattern JSON:', this.patternConfigJSON);
        
        this.generateConfigurationForm(patternType);
        
        console.log('🖼️ Calling updateModalPreview...');
        this.updateModalPreview();
        this.updateModalDescription();
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
        
        // Generate form fields - one parameter per row for better horizontal layout
        for (let i = 0; i < paramList.length; i++) {
            const paramConfig = paramList[i];
            html += this.generateParameterField(paramConfig.name, paramConfig);
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
        console.log('🎨 generateParameterField called with horizontal layout for:', paramName);
        const fieldId = `modal-${this.config.patternType}-${paramName.replace(/_/g, '-')}`;
        
        let html = `<div class="mb-3">`;
        
        // Use horizontal layout: label and input on same row
        html += `<div class="row align-items-center">`;
        html += `<div class="col-sm-5">`;
        html += `<label for="${fieldId}" class="form-label mb-0">${paramConfig.label || paramName}</label>`;
        html += `</div>`;
        html += `<div class="col-sm-7">`;
        
        console.log('🔧 Building horizontal form field HTML for:', paramName);
        
        if (paramConfig.options && Array.isArray(paramConfig.options)) {
            // Dropdown for options
            html += `<select class="form-select form-select-sm" id="${fieldId}" data-param="${paramName}">`;
            for (const option of paramConfig.options) {
                const selected = option.value == paramConfig.default ? 'selected' : '';
                html += `<option value="${option.value}" ${selected}>${option.label}</option>`;
            }
            html += '</select>';
        } else if (paramConfig.input_type === 'select' && paramConfig.options) {
            // Handle input_type: select for dropdown elements
            html += `<select class="form-select form-select-sm" id="${fieldId}" data-param="${paramName}">`;
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
            
            html += `<input type="${inputType}" class="form-control form-control-sm" id="${fieldId}" 
                     data-param="${paramName}" value="${paramConfig.default}" 
                     step="${step}" min="${min}" max="${max}">`;
        }
        
        html += `</div>`; // End col-sm-7
        html += `</div>`; // End row
        
        // Description below the row
        if (paramConfig.description) {
            html += `<div class="form-text mt-1 small text-muted">${paramConfig.description}</div>`;
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
                this.updateModalDescription();
            });
        });
    }
    
    onModalShow() {
        console.log('Modal shown, loading current config');
        this.loadConfigToModal();
        this.updateModalPreview();
        this.updateModalDescription();
    }
    
    async loadFromSession() {
        try {
            const saved = sessionStorage.getItem('chessboard_pattern_json');
            if (saved) {
                this.patternConfigJSON = JSON.parse(saved);
                console.log('✅ Loaded pattern JSON from session:', this.patternConfigJSON);
                return;
            }
        } catch (error) {
            console.warn('Could not load pattern JSON from session:', error);
        }
        
        // If no session data, get default pattern from API
        await this.loadDefaultPatternJSON();
    }
    
    async loadDefaultPatternJSON() {
        try {
            console.log('🎯 Getting default pattern configuration from API...');
            const response = await fetch('/api/default_pattern_config');
            const data = await response.json();
            
            if (data.success) {
                this.patternConfigJSON = data.pattern_config;
                console.log('✅ Loaded default pattern JSON:', this.patternConfigJSON);
            } else {
                throw new Error(data.error || 'Failed to get default pattern');
            }
        } catch (error) {
            console.error('❌ Failed to load default pattern:', error);
            // Fallback to hardcoded default
            this.patternConfigJSON = {
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
            console.log('🔧 Using fallback default pattern:', this.patternConfigJSON);
        }
    }

    saveToSession() {
        try {
            if (this.patternConfigJSON) {
                sessionStorage.setItem('chessboard_pattern_json', JSON.stringify(this.patternConfigJSON));
                console.log('💾 Saved pattern JSON to session');
            }
        } catch (error) {
            console.warn('Could not save pattern JSON to session:', error);
        }
    }
    
    loadConfigToModal() {
        console.log('loadConfigToModal called with patternConfigJSON:', this.patternConfigJSON);
        
        // Since we're using the PatternSelectionModal class, this method might not be needed
        // But if we need to load current config into modal, we'll work with JSON
        if (!this.patternConfigJSON || !this.patternConfigJSON.pattern_id) {
            console.log('No pattern configuration JSON available for modal');
            return;
        }
        
        // Set pattern type dropdown if it exists
        const patternTypeSelect = document.getElementById('pattern-type-select');
        if (patternTypeSelect) {
            patternTypeSelect.value = this.patternConfigJSON.pattern_id;
            console.log('Set pattern type dropdown to:', this.patternConfigJSON.pattern_id);
        }
        
        // The modal should handle its own parameter loading via PatternSelectionModal
        console.log('Modal parameter loading delegated to PatternSelectionModal class');
    }
    
    applyConfiguration() {
        console.log('🔧 Applying configuration with patternConfigJSON:', this.patternConfigJSON);
        
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
    
    /**
     * Set pattern configuration from complete JSON
     */
    setPatternConfigJSON(patternConfigJSON) {
        console.log('🔧 Setting pattern config JSON:', patternConfigJSON);
        this.patternConfigJSON = patternConfigJSON;
        
        // Save to session storage
        this.saveToSession();
        
        // Update displays
        this.updateDisplay();
        
        console.log('✅ Pattern config JSON set successfully');
    }
    
    updateDisplay() {
        // Update the main chessboard card display
        this.updatePatternImage();
        this.updatePatternDescription();
    }
    
    updatePatternImage() {
        console.log('🖼️ updatePatternImage called');
        console.log('📊 Current patternConfigJSON:', this.patternConfigJSON);
        
        const img = document.getElementById('chessboard-preview-img');
        console.log('🎯 Preview img element:', img);
        
        if (!img) {
            console.warn('⚠️ Cannot update pattern image - img element not found');
            return;
        }
        
        const patternJSON = this.getPatternJSON();
        if (!patternJSON) {
            console.warn('⚠️ Cannot update pattern image - no pattern configuration');
            img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIENvbmZpZ3VyYXRpb248L3RleHQ+PC9zdmc+';
            return;
        }
        
        const params = {
            pattern_json: JSON.stringify(patternJSON),
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
        
        if (!typeDisplay || !specsDisplay) return;
        
        const patternJSON = this.getPatternJSON();
        if (!patternJSON) {
            typeDisplay.textContent = 'No Pattern';
            specsDisplay.textContent = 'Please configure a pattern';
            return;
        }
        
        // Use the pattern name from the JSON configuration
        typeDisplay.textContent = patternJSON.name || 'Unknown Pattern';
        
        // Generate specs display from pattern parameters
        let specs = [];
        if (patternJSON.parameters) {
            // Try to get parameter metadata from API if available
            const patternConfig = this.patternConfigurations[patternJSON.pattern_id];
            
            for (const [paramName, value] of Object.entries(patternJSON.parameters)) {
                let displayValue = value;
                let label = paramName;
                
                // Try to get nice label from API metadata
                if (patternConfig && patternConfig.parameters) {
                    const paramConfig = patternConfig.parameters[paramName];
                    if (paramConfig) {
                        label = paramConfig.label || paramName;
                        
                        // Format value based on parameter type
                        if (paramConfig.type === 'float' && typeof value === 'number') {
                            displayValue = value.toFixed(3);
                            // Add unit if it's a size parameter
                            if (paramName.includes('size')) {
                                displayValue += ' m';
                            }
                        }
                    }
                }
                
                specs.push(`${label}: ${displayValue}`);
            }
        }
        
        specsDisplay.textContent = specs.length > 0 ? specs.join(', ') : 'No parameters configured';
    }
    
    updateModalPreview() {
        console.log('🖼️ updateModalPreview called');
        console.log('📊 Current patternConfigJSON:', this.patternConfigJSON);
        
        // Try to use the new modal structure first
        const previewContainer = document.getElementById('pattern-preview-container');
        const legacyImg = document.getElementById('chessboard-modal-preview');
        
        console.log('🎯 Preview container:', previewContainer);
        console.log('🎯 Legacy img element:', legacyImg);
        
        if (!previewContainer && !legacyImg) {
            console.warn('⚠️ Cannot update modal preview - no preview elements found');
            return;
        }
        
        const patternJSON = this.getPatternJSON();
        if (!patternJSON) {
            console.warn('⚠️ Cannot update modal preview - no pattern configuration');
            const noImageSrc = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIENvbmZpZ3VyYXRpb248L3RleHQ+PC9zdmc+';
            
            if (previewContainer) {
                previewContainer.innerHTML = `<img src="${noImageSrc}" alt="No Configuration" class="img-fluid border rounded" style="max-height: 200px;">`;
            } else if (legacyImg) {
                legacyImg.src = noImageSrc;
            }
            return;
        }
        
        const params = {
            pattern_json: JSON.stringify(patternJSON),
            pixel_per_square: 20,
            border_pixels: 0,
            t: Date.now() // Cache busting
        };
        
        const url = `/api/pattern_image?${new URLSearchParams(params)}`;
        console.log('🔗 Generated URL (with cache-busting):', url);
        
        if (previewContainer) {
            // Use new modal structure
            previewContainer.innerHTML = `
                <img src="${url}" 
                     alt="Pattern Preview" 
                     class="img-fluid border rounded" 
                     style="max-height: 200px;"
                     onload="console.log('✅ Modal preview image loaded successfully')"
                     onerror="console.error('❌ Failed to load modal preview image:', '${url}'); this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIEltYWdlPC90ZXh0Pjwvc3ZnPg==';">
            `;
        } else if (legacyImg) {
            // Use legacy img element
            legacyImg.src = url;
            legacyImg.onload = () => {
                console.log('✅ Modal preview image loaded successfully');
            };
            legacyImg.onerror = () => {
                console.error('❌ Failed to load modal preview image:', url);
            };
        }
    }
    
    updateModalDescription() {
        console.log('📝 updateModalDescription called');
        const infoDisplay = document.getElementById('pattern-info-display');
        
        if (!infoDisplay) {
            console.log('⚠️ Cannot update modal description - element not found');
            return;
        }
        
        const patternJSON = this.getPatternJSON();
        if (!patternJSON) {
            console.log('⚠️ Cannot update modal description - no pattern configuration');
            infoDisplay.innerHTML = '<em>No pattern configured</em>';
            return;
        }
        
        const patternConfig = this.patternConfigurations[patternJSON.pattern_id];
        if (!patternConfig) {
            console.log('⚠️ Cannot update modal description - no pattern config found in API');
            infoDisplay.innerHTML = `
                <div class="pattern-summary">
                    <div class="mb-2"><strong>${patternJSON.name || 'Unknown Pattern'}</strong></div>
                    <div class="small text-muted"><em>No metadata available</em></div>
                </div>
            `;
            return;
        }
        
        // Build description from current parameters
        let descriptionParts = [];
        
        if (patternJSON.parameters) {
            for (const [paramName, value] of Object.entries(patternJSON.parameters)) {
                // Find parameter config - handle both array and object formats
                let paramConfig = null;
                
                if (Array.isArray(patternConfig.parameters)) {
                    paramConfig = patternConfig.parameters.find(p => p.name === paramName);
                } else {
                    paramConfig = patternConfig.parameters[paramName];
                }
                
                if (paramConfig) {
                    const label = paramConfig.label || paramName;
                    // Format value based on parameter type
                    let displayValue = value;
                    if (paramConfig.type === 'float' && typeof value === 'number') {
                        displayValue = value.toFixed(3);
                        // Add unit if it's a size parameter
                        if (paramName.includes('size')) {
                            displayValue += ' m';
                        }
                    }
                    descriptionParts.push(`<strong>${label}:</strong> ${displayValue}`);
                }
            }
        }
        
        // Update the display
        const descriptionHTML = descriptionParts.length > 0 
            ? descriptionParts.join('<br>') 
            : '<em>No parameters configured</em>';
            
        infoDisplay.innerHTML = `
            <div class="pattern-summary">
                <div class="mb-2"><strong>${patternConfig.name}</strong></div>
                <div class="small text-muted">${descriptionHTML}</div>
            </div>
        `;
        
        console.log('📝 Updated modal description:', descriptionParts.join(', '));
    }
    
    // Create a valid JSON pattern object for the backend
    getPatternJSON() {
        // If we already have a complete JSON configuration, return it directly
        if (this.patternConfigJSON && typeof this.patternConfigJSON === 'object') {
            console.log('🎯 Using existing patternConfigJSON:', this.patternConfigJSON);
            return this.patternConfigJSON;
        }
        
        console.log('⚠️ No patternConfigJSON available, returning null');
        return null;
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChessboardConfig;
}
