// Controls Module for LATTE Viewer

class ControlsManager {
    constructor(dataManager, treeRenderer, annotationSystem) {
        this.dataManager = dataManager;
        this.treeRenderer = treeRenderer;
        this.annotationSystem = annotationSystem;
        this.showMask = false;
        this.showAnnotations = false;
        this.maxLevel = 0;
        this.currentLevel = 0;
        this.currentSearchTerm = '';
        this.matchingClusters = new Set();
    }
    
    // Initialize event listeners
    init() {
        this.setupFileInput();
        this.setupIconToggles();
        this.setupLevelSlider();
        this.setupSearch();
        this.displayMetadata();
        
        // Apply initial states
        this.updateToggleVisuals();
    }
    
    // Setup file input handling
    setupFileInput() {
        const fileInput = document.getElementById('jsonFile');
        if (fileInput) {
            fileInput.addEventListener('change', (event) => {
                const file = event.target.files[0];
                if (file) {
                    // Reset everything before loading new file
                    this.resetVisualization();
                    
                    this.dataManager.loadFromFile(file)
                        .then(() => {
                            this.displayMetadata();
                            this.updateVisualization();
                        })
                        .catch(error => {
                            alert(error.message);
                        });
                }
            });
        }
    }
    
    // Setup icon toggle controls
    setupIconToggles() {
        const maskToggle = document.getElementById('showMaskToggle');
        const annotationsToggle = document.getElementById('showAnnotationsToggle');
        
        if (maskToggle) {
            maskToggle.addEventListener('click', (e) => {
                e.preventDefault();
                this.showMask = !this.showMask;
                this.updateToggleVisuals();
                this.updateVisualization();
            });
        }
        
        if (annotationsToggle) {
            annotationsToggle.addEventListener('click', (e) => {
                e.preventDefault();
                this.showAnnotations = !this.showAnnotations;
                this.updateToggleVisuals();
                this.updateVisualization();
            });
        }
    }
    
    // Setup level slider
    setupLevelSlider() {
        const levelSlider = document.getElementById('levelSlider');
        
        if (levelSlider) {
            levelSlider.addEventListener('input', (e) => {
                this.currentLevel = parseInt(e.target.value);
                this.updateVisualization();
            });
        }
    }

    // Setup search functionality
    setupSearch() {
        const searchInput = document.getElementById('searchInput');
        const clearSearch = document.getElementById('clearSearch');
        
        if (searchInput) {
            // Handle search input with debouncing
            let searchTimeout;
            searchInput.addEventListener('input', (e) => {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    this.handleSearch(e.target.value);
                }, 300); // 300ms debounce
            });

            // Handle Enter key for immediate search
            searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    clearTimeout(searchTimeout);
                    this.handleSearch(e.target.value);
                }
            });
        }

        if (clearSearch) {
            clearSearch.addEventListener('click', () => {
                if (searchInput) {
                    searchInput.value = '';
                    this.handleSearch('');
                }
            });
        }
    }
    
    // Update visual state of toggle icons
    updateToggleVisuals() {
        const maskToggle = document.getElementById('showMaskToggle');
        const annotationsToggle = document.getElementById('showAnnotationsToggle');
        
        if (maskToggle) {
            if (this.showMask) {
                maskToggle.classList.add('active');
            } else {
                maskToggle.classList.remove('active');
            }
        }
        
        if (annotationsToggle) {
            if (this.showAnnotations) {
                annotationsToggle.classList.add('active');
            } else {
                annotationsToggle.classList.remove('active');
            }
        }
    }
    
    // Handle mask toggle changes
    onMaskToggle(checked) {
        this.showMask = checked;
        this.updateToggleVisuals();
        this.updateVisualization();
    }
    
    // Handle annotation toggle changes
    onAnnotationToggle(checked) {
        this.showAnnotations = checked;
        this.updateToggleVisuals();
        this.updateVisualization();
    }
    
    // Handle search input
    handleSearch(searchTerm) {
        this.currentSearchTerm = searchTerm.trim();
        
        // Update clear button visibility
        const clearSearch = document.getElementById('clearSearch');
        if (clearSearch) {
            clearSearch.style.display = this.currentSearchTerm ? 'flex' : 'none';
        }

        if (this.currentSearchTerm) {
            // Perform search
            const searchResults = this.dataManager.searchClusters(this.currentSearchTerm);
            this.matchingClusters = searchResults.matchingClusters;
        } else {
            // Clear search
            this.matchingClusters = new Set();
        }

        this.updateVisualization();
        
        // Refresh sidebar with new search term if a cluster is currently displayed
        if (window.latteViewer && window.latteViewer.sidebar) {
            window.latteViewer.sidebar.refreshWithSearch(this.currentSearchTerm);
        }
    }

    // Update visualization based on current state
    updateVisualization() {
        if (!this.dataManager.hasData()) {
            return;
        }
        
        // Render tree once (will skip if already rendered)
        this.treeRenderer.render();
        
        // Update tree state using CSS classes
        this.treeRenderer.updateNodeColors(this.showMask);
        this.treeRenderer.filterByLevel(this.currentLevel);
        this.treeRenderer.toggleAnnotations(this.showAnnotations);
        
        // Apply search highlighting
        this.treeRenderer.highlightSearch(this.matchingClusters);
        
        // Update annotations display (render once, then filter)
        this.annotationSystem.showAll(this.currentLevel);
        this.annotationSystem.toggle(this.showAnnotations, this.currentLevel);
    }
    
    // Display metadata information
    displayMetadata() {
        const metadata = this.dataManager.getMetadata();
        if (!metadata) {
            d3.select("#metadata").style("display", "none");
            d3.select(".level-slider-container").style("display", "none");
            d3.select(".search-container").style("display", "none");
            return;
        }
        
        // Update max level and slider
        this.maxLevel = metadata.max_level;
        this.currentLevel = 0; // Start showing all levels (most detailed)
        this.updateLevelSlider();
        
        // Show search container
        d3.select(".search-container").style("display", "flex");
        
        const metadataDiv = d3.select("#metadata");
        metadataDiv.style("display", "block")
            .html(`
                <strong>Dataset:</strong> 
                ${metadata.total_documents} docs, 
                ${metadata.total_clusters} clusters, 
                ${metadata.max_level + 1} levels
                ${metadata.mask_baseline_proportion ? 
                    ` • Baseline: ${(metadata.mask_baseline_proportion * 100).toFixed(1)}%` : ''}
            `);
    }
    
    // Update level slider based on current dataset
    updateLevelSlider() {
        const levelSlider = document.getElementById('levelSlider');
        const sliderContainer = document.querySelector('.level-slider-container');
        
        if (levelSlider && sliderContainer) {
            levelSlider.min = 0;
            levelSlider.max = this.maxLevel;
            levelSlider.value = 0; // Start at most detailed level
            sliderContainer.style.display = 'flex';
        }
    }
    
    // Load embedded data (for development/testing)
    loadEmbeddedData(embeddedData) {
        if (this.dataManager.loadEmbeddedData(embeddedData)) {
            this.displayMetadata();
            this.updateVisualization();
            return true;
        }
        return false;
    }
    
    // Get current state for external access
    getState() {
        return {
            showMask: this.showMask,
            showAnnotations: this.showAnnotations
        };
    }
    
    // Set state programmatically
    setState(state) {
        if (state.showMask !== undefined) {
            this.showMask = state.showMask;
        }
        if (state.showAnnotations !== undefined) {
            this.showAnnotations = state.showAnnotations;
        }
        this.updateToggleVisuals();
        this.updateVisualization();
    }
    
    // Reset visualization for new file loading
    resetVisualization() {
        // Clear tree renderer
        this.treeRenderer.reset();
        
        // Clear annotation system
        this.annotationSystem.reset();
        
        // Clear sidebar
        if (window.latteViewer && window.latteViewer.sidebar) {
            window.latteViewer.sidebar.renderEmptyState();
        }
        
        // Reset control state
        this.showMask = false;
        this.showAnnotations = false;
        this.maxLevel = 0;
        this.currentLevel = 0;
        this.currentSearchTerm = '';
        this.matchingClusters = new Set();
        
        // Clear search input
        const searchInput = document.getElementById('searchInput');
        const clearSearch = document.getElementById('clearSearch');
        if (searchInput) {
            searchInput.value = '';
        }
        if (clearSearch) {
            clearSearch.style.display = 'none';
        }
        
        // Update UI
        this.updateToggleVisuals();
        
        // Hide metadata, slider, and search until new data loads
        d3.select("#metadata").style("display", "none");
        d3.select(".level-slider-container").style("display", "none");
        d3.select(".search-container").style("display", "none");
    }
} 